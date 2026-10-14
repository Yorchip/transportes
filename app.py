import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# Configuración de la página
st.set_page_config(
    page_title="IA Legal - RAG Megaleyes", 
    page_icon="⚖️", 
    layout="centered"
)

st.title("⚖️ Asistente Jurídico de Transportes Inteligente (RAG)")
st.markdown("Pregúntame sobre transporte y seguridad vial. Las respuestas se generan mediante IA basándose estrictamente en los textos oficiales.")

# Configurar la API de Gemini usando los secrets de Streamlit (seguro para GitHub)
# O puedes poner tu clave directamente entre comillas para pruebas en local
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "TU_API_KEY_DE_GOOGLE" # Pégala aquí solo para pruebas locales

genai.configure(api_key=GOOGLE_API_KEY)

# Usamos un modelo rápido y eficiente de Gemini
response = cliente_gemini.models.generate_content(
    model="gemini-3.5-flash-lite",  # O "gemini-1.5-flash"
    contents=prompt
)

respuesta_texto = response.text

# Inicializar ChromaDB de forma persistente
@st.cache_resource
def cargar_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    collection = client.get_or_create_collection(
        name="leyes_rag",
        embedding_function=multilingual_ef
    )
    return collection

try:
    collection = cargar_chroma()
except Exception as e:
    st.error(f"Error al cargar la base de datos vectorial: {e}")
    st.stop()

# Caja de chat o de pregunta
pregunta = st.text_input(
    "¿Qué consulta legal deseas realizar?", 
    placeholder="Ej: ¿Cuáles son las sanciones por exceso de peso según la LOTT?"
)

if pregunta:
    with st.spinner("Buscando en las megaleyes y consultando a la IA..."):
        # 1. Buscar los fragmentos más relevantes en ChromaDB
        resultados = collection.query(
            query_texts=[pregunta],
            n_results=3
        )
        
        contexto_textos = ""
        documentos_encontrados = []
        
        if resultados and resultados['documents']:
            documentos = resultados['documents'][0]
            metadatos = resultados['metadatas'][0]
            
            for i, doc in enumerate(documentos):
                meta = metadatos[i]
                contexto_textos += f"\n--- [{meta.get('titulo', 'Artículo')} | {meta.get('normativa', '')}] ---\n{doc}\n"
                documentos_encontrados.append((meta, doc))

        # 2. Construir el prompt estricto anti-alucinaciones para Gemini
        prompt_sistema = f"""
Eres un asistente jurídico experto. Responde a la pregunta del usuario utilizando ÚNICAMENTE la información contenida en los fragmentos legales oficiales que se proporcionan a continuación. 
Si la respuesta no está en los textos, di claramente que la información no aparece en las leyes facilitadas. No inventes datos. Cita los artículos correspondientes en tu respuesta.

TEXTOS LEGALES DE REFERENCIA:
{contexto_textos}

PREGUNTA DEL USUARIO:
{pregunta}
        """

        # 3. Generar la respuesta con Gemini
        try:
            respuesta_gemini = modelo_ia.generate_content(prompt_sistema)
            
            st.subheader("🤖 Respuesta de la IA:")
            st.markdown(respuesta_gemini.text)
            
            # Mostrar los artículos de origen para total transparencia
            with st.expander("📚 Ver artículos legales originales consultados"):
                for meta, doc in documentos_encontrados:
                    st.markdown(f"**📌 {meta.get('titulo', '')}** ({meta.get('normativa', '')})")
                    st.markdown(f"> {doc}")
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"Error al generar la respuesta con la API de Google: {e}")
