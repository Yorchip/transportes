import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

st.set_page_config(page_title="Megaleyes - Consulta Legal", layout="wide")
st.title("⚖️ Megaleyes: Asistente Legal RAG")

@st.cache_resource
def inicializar_recursos():
    # 1. Cliente Gemini con el nuevo SDK oficial
    cliente_gemini = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=HttpOptions(api_version="v1")
    )
    
    # 2. Conexión segura a la base de datos persistente ChromaDB
    cliente_chroma = chromadb.PersistentClient(path="chroma_db")
    
    # Conectamos con la colección confirmada: 'leyes_rag'
    try:
        coleccion = cliente_chroma.get_collection(name="leyes_rag")
    except Exception:
        coleccion = cliente_chroma.create_collection(name="leyes_rag")
        
    # 3. Carga del modelo local de embeddings
    modelo_embed = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    return cliente_gemini, coleccion, modelo_embed

try:
    cliente_gemini, coleccion, modelo_embed = inicializar_recursos()
    
    # Indicador de estado en la barra lateral
    total_docs = coleccion.count()
    st.sidebar.success(f" Base de datos: {total_docs} fragmentos disponibles.")
except Exception as e:
    st.error(f"Error al inicializar los recursos: {e}")
    st.stop()

# Interfaz de consulta
consulta_usuario = st.text_input("Haz tu consulta legal:")

if consulta_usuario:
    with st.spinner("Buscando en la normativa y redactando respuesta..."):
        try:
            # 1. Generar embedding vectorial de la pregunta localmente
            vector_consulta = modelo_embed.encode(consulta_usuario).tolist()

            # 2. Búsqueda semántica en ChromaDB (Recuperando 8 fragmentos para mayor cobertura)
            resultados = coleccion.query(
                query_embeddings=[vector_consulta],
                n_results=8
            )

            # Extraer textos encontrados
            documentos_recuperados = resultados['documents'][0] if resultados['documents'] else []
            
            if not documentos_recuperados:
                st.warning("No se han recuperado textos relevantes.")
            else:
                contexto = "\n\n---\n\n".join(documentos_recuperados)

                # 3. Prompt estricto anti-alucinaciones
                prompt = f"""
Eres un asistente legal extremadamente riguroso. Tu única fuente de verdad es el Contexto Legal proporcionado a continuación.

REGLAS ESTRICTAS:
1. Responde únicamente utilizando la información explícita que aparece en el contexto.
2. No inventes artículos, plazos, multas ni supuestos que no estén escritos textualmente.
3. Si la respuesta exacta no se encuentra en el contexto proporcionado, responde obligatoriamente y sin añadir nada más: "No lo he visto".

Contexto Legal:
{contexto}

Pregunta del usuario:
{consulta_usuario}

Respuesta:
"""

                # 4. Generación con Gemini 3.5 Flash-Lite (Temperatura 0.0)
                respuesta = cliente_gemini.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0
                    )
                )

                # Mostrar respuesta
                st.markdown("### Respuesta:")
                st.write(respuesta.text)

                # Desplegable para auditoría / depuración de fragmentos
                with st.expander("🔍 Ver los 8 fragmentos recuperados de la base de datos"):
                    for i, doc in enumerate(documentos_recuperados, 1):
                        st.markdown(f"**Fragmento {i}:**")
                        st.write(doc)
                        st.markdown("---")

        except Exception as e:
            st.error(f"Error procesando la consulta: {e}")
