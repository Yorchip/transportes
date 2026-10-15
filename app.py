import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

st.set_page_config(page_title="Megaleyes - Consulta Legal", layout="wide")
st.title("⚖️ Megaleyes: Asistente Legal RAG")

@st.cache_resource
def inicializar_recursos():
    fn_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    cliente_chroma = chromadb.PersistentClient(path="chroma_db")
    coleccion = cliente_chroma.get_collection(
        name="leyes_rag", 
        embedding_function=fn_embedding
    )
    cliente_gemini = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=HttpOptions(api_version="v1")
    )
    return cliente_gemini, coleccion

try:
    cliente_gemini, coleccion = inicializar_recursos()
    total_docs = coleccion.count()
    st.sidebar.success(f"Base de datos: {total_docs} fragmentos disponibles.")
except Exception as e:
    st.error(f"Error al inicializar los recursos: {e}")
    st.stop()

consulta_usuario = st.text_input("Haz tu consulta legal (ej: código, multa, concepto):")

if consulta_usuario:
    with st.spinner("Buscando en profundidad en la normativa..."):
        try:
            # Subimos a 25 resultados para asegurar que el fragmento correcto entra sí o sí
            resultados = coleccion.query(
                query_texts=[consulta_usuario],
                n_results=25
            )

            documentos_recuperados = resultados['documents'][0] if resultados['documents'] else []
            
            if not documentos_recuperados:
                st.warning("No se ha encontrado nada en la base de datos.")
            else:
                contexto = "\n\n---\n\n".join(documentos_recuperados)

                # Prompt diseñado para que Gemini filtre entre la gran cantidad de fragmentos
                prompt = f"""
Eres un asistente legal de precisión. Tienes a continuación un conjunto amplio de fragmentos legales recuperados. Tu misión es buscar con lupa dentro de ellos el código, artículo o infracción exacto que responde a la consulta del usuario.

REGLAS:
1. Revisa minuciosamente todos los fragmentos. Si encuentras la infracción o concepto solicitado, responde estrictamente con este formato esquemático y sin rodeos:
   - **Código:** [Código de la infracción si aparece, ej: AA01.05]
   - **Concepto:** [Descripción breve]
   - **Normas Infringidas:** [Artículos o normas aplicables]
   - **Cuantía:** [Importe si consta]
   - **Inmovilización:** [Indica si procede o no]
2. Si tras revisar exhaustivamente todos los fragmentos la respuesta exacta NO está en ninguno de ellos, responde obligatoriamente y sin añadir nada más: "No lo he visto".
3. No inventes ningún dato que no esté escrito textualmente en los textos.

Contexto Legal:
{contexto}

Consulta del usuario:
{consulta_usuario}

Respuesta:
"""

                respuesta = cliente_gemini.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2  # Bajamos a 0.2 para que sea preciso filtrando entre tanto texto
                    )
                )

                st.markdown("### Respuesta:")
                st.write(respuesta.text)

                with st.expander("🔍 Ver los 25 fragmentos analizados"):
                    for i, doc in enumerate(documentos_recuperados, 1):
                        st.markdown(f"**Fragmento {i}:**")
                        st.write(doc)
                        st.markdown("---")

        except Exception as e:
            st.error(f"Error procesando la consulta: {e}")
