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

# 1. PRIMERO DEFINIMOS EL CAMPO DE TEXTO
consulta_usuario = st.text_input("Haz tu consulta legal:")

# 2. DESPUÉS COMPROBAMOS SI EL USUARIO HA ESCRITO ALGO
if consulta_usuario:
    with st.spinner("Buscando en la normativa y redactando respuesta..."):
        try:
            resultados = coleccion.query(
                query_texts=[consulta_usuario],
                n_results=8
            )

            documentos_recuperados = resultados['documents'][0] if resultados['documents'] else []
            
            if not documentos_recuperados:
                st.warning("No se han recuperado textos relevantes.")
            else:
                contexto = "\n\n---\n\n".join(documentos_recuperados)

                prompt = f"""
Eres un asistente legal analítico y conciso. Tu objetivo es responder la consulta relacionando los conceptos del Contexto Legal proporcionado.

REGLAS DE RESPUESTA:
1. Responde de forma clara y directísima, utilizando oraciones cortas y puntos clave.
2. Sintetiza y conecta los conceptos, artículos o plazos presentes en el contexto para dar una respuesta completa.
3. Bástate exclusivamente en la información contenida en el contexto. No asumas ni agregues datos externos.
4. Si el contexto no contiene información suficiente para responder a la consulta, responde únicamente: "No lo he visto".

Contexto Legal:
{contexto}

Consulta del usuario:
{consulta_usuario}

Respuesta concisa:
"""

                respuesta = cliente_gemini.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2
                    )
                )

                st.markdown("### Respuesta:")
                st.write(respuesta.text)

                with st.expander("🔍 Ver fragmentos recuperados"):
                    for i, doc in enumerate(documentos_recuperados, 1):
                        st.markdown(f"**Fragmento {i}:**")
                        st.write(doc)
                        st.markdown("---")

        except Exception as e:
            st.error(f"Error procesando la consulta: {e}")
