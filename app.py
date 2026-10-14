import streamlit as st
from google import genai
from google.genai.types import HttpOptions
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Configuración de la página de Streamlit
st.set_page_config(
    page_title="Megaleyes - Buscador Legal RAG",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Megaleyes: Asistente de Consulta Legal")
st.write("Realiza preguntas sobre las normativas y artículos legales indexados.")

# 2. Inicializar cliente de Google GenAI y base de datos con caché
@st.cache_resource
def inicializar_recursos():
    # Inicializa el cliente moderno compatible con los nuevos tokens
    cliente = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=HttpOptions(api_version="v1")
    )
    
    # Cargar modelo de embeddings local
    modelo_embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Conectar con ChromaDB local
    cliente_chroma = chromadb.PersistentClient(path="chroma_db")
    coleccion = cliente_chroma.get_collection(name="leyes_megaleyes")
    
    return cliente, modelo_embeddings, coleccion

try:
    cliente_gemini, modelo_embeddings, coleccion = inicializar_recursos()
except Exception as e:
    st.error(f"Error al inicializar los recursos o la base de datos: {e}")
    st.stop()

# 3. Interfaz de usuario para la consulta
consulta_usuario = st.text_input("¿Qué consulta legal deseas realizar?", placeholder="Ej: ¿Cuáles son las sanciones por...?")

if consulta_usuario:
    with st.spinner("Buscando en los documentos legales y generando respuesta..."):
        try:
            # Generar embedding de la consulta del usuario
            vector_consulta = modelo_embeddings.encode(consulta_usuario).tolist()
            
            # Buscar fragmentos relevantes en ChromaDB
            resultados = coleccion.query(
                query_embeddings=[vector_consulta],
                n_results=3
            )
            
            documentos_recuperados = resultados['documents'][0] if resultados['documents'] else []
            contexto = "\n\n---\n\n".join(documentos_recuperados)
            
            # Construcción del prompt con contexto legal
            prompt = f"""
Eres un asistente legal experto. Utiliza exclusivamente el siguiente contexto legal para responder a la pregunta del usuario. Si la respuesta no está en el contexto, indícalo claramente.

Contexto Legal:
{contexto}

Pregunta del usuario:
{consulta_usuario}

Respuesta profesional y fundamentada:
"""
            
            # Generación de respuesta usando el cliente moderno
            response = cliente_gemini.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )
            
            # Mostrar resultado en pantalla
            st.markdown("### Respuesta:")
            st.write(response.text)
            
            with st.expander("Ver fuentes legales consultadas"):
                for i, doc in enumerate(documentos_recuperados, 1):
                    st.markdown(f"**Fragmento {i}:**")
                    st.write(doc)
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"Error al generar la respuesta con la API de Google: {e}")
