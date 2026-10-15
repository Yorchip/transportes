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

consulta_usuario = st.text_input("Haz tu consulta legal:")

if consulta_usuario:
    with st.spinner("Buscando en la normativa..."):
        try:
            # Ampliamos a 15 fragmentos para ampliar el margen de acierto
            resultados = coleccion.query(
                query_texts=[consulta_usuario],
                n_results=15
            )

            documentos_recuperados = resultados['documents'][0] if resultados['documents'] else []
            
            # Mostramos SIEMPRE los fragmentos arriba para saber qué está leyendo
            with st.expander("🔍 Fragmentos recuperados por ChromaDB (Abre para comprobar)"):
                for i, doc in enumerate(documentos_recuperados, 1):
                    st.markdown(f"**Fragmento {i}:**")
                    st.write(doc)
                    st.markdown("---")

            if not documentos_recuperados:
                st.warning("ChromaDB no devolvió ningún fragmento para esta búsqueda.")
            else:
                contexto = "\n\n---\n\n".join(documentos_recuperados)

                prompt = f"""
Eres un asistente legal experto. Tu objetivo es responder la consulta basándote exclusivamente en el Contexto Legal proporcionado.

INSTRUCCIONES:
1. Analiza con atención todos los fragmentos del contexto.
2. Si la consulta involucra una infracción o sanción mencionada o deducible del contexto, redacta la respuesta usando este esquema conciso:
   - **Código:** [Indica el código si aparece o la referencia]
   - **Concepto:** [Descripción breve]
   - **Normas Infringidas:** [Artículos o normas aplicables]
   - **Cuantía:** [Importe o multa si consta]
   - **Inmovilización:** [Indica si procede o no según el contexto]
3. Si la respuesta a la pregunta no se encuentra implícita ni explícitamente en el contexto, responde únicamente: "No lo he visto".

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
                        temperature=0.3
                    )
                )

                st.markdown("### Respuesta:")
                st.write(respuesta.text)

        except Exception as e:
            st.error(f"Error procesando la consulta: {e}")
