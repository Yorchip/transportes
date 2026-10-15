import streamlit as st
import glob
import os
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

st.set_page_config(page_title="Megaleyes - Consulta Legal", layout="wide")
st.title("⚖️ Megaleyes: Asistente Legal RAG")

@st.cache_resource
def cargar_textos():
    # Lee todos los archivos .md y .txt de la carpeta 'leyes'
    archivos = glob.glob("leyes/*.md") + glob.glob("leyes/*.txt")
    bloques_totales = []
    
    for ruta in archivos:
        with open(ruta, "r", encoding="utf-8") as f:
            nombre_archivo = os.path.basename(ruta)
            contenido = f.read()
            # Dividimos el archivo en fragmentos por párrafos o artículos para buscar con precisión
            parrafos = contenido.split("\n\n")
            for p in parrafos:
                if p.strip():
                    bloques_totales.append({"origen": nombre_archivo, "texto": p})
                    
    return bloques_totales, len(archivos)

@st.cache_resource
def inicializar_gemini():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=HttpOptions(api_version="v1")
    )

try:
    base_conocimiento, num_archivos = cargar_textos()
    cliente_gemini = inicializar_gemini()
    st.sidebar.success(f"Base de conocimiento: {num_archivos} archivos cargados.")
except Exception as e:
    st.error(f"Error cargando los archivos: {e}")
    st.stop()

consulta_usuario = st.text_input("Haz tu consulta legal:")

if consulta_usuario:
    with st.spinner("Buscando en la normativa..."):
        try:
            # Búsqueda inteligente en Python: filtramos fragmentos que contengan alguna palabra de la consulta
            palabras_clave = [w.lower() for w in consulta_usuario.split() if len(w) > 2]
            
            fragmentos_relevantes = []
            for bloque in base_conocimiento:
                texto_lower = bloque["texto"].lower()
                # Si coincide alguna palabra clave importante o código, seleccionamos el fragmento
                if any(palabra in texto_lower for palabra in palabras_clave):
                    fragmentos_relevantes.append(f"[{bloque['origen']}]\n{bloque['texto']}")
            
            # Si no encuentra coincidencia exacta por palabra, cogemos una muestra general para no fallar
            if not fragmentos_relevantes:
                fragmentos_relevantes = [f"[{b['origen']}]\n{b['texto']}" for b in base_conocimiento[:30]]
            
            # Limitamos a los fragmentos más directos (máximo 15 para cuidar los tokens)
            contexto_filtrado = "\n\n---\n\n".join(fragmentos_relevantes[:15])

            # Prompt con tus reglas exactas
            prompt = f"""
Eres un asistente legal experto. Tienes a continuación fragmentos de la normativa legal relacionados con la consulta.

REGLAS DE RESPUESTA:
1. Sé extremadamente directo y conciso. Evita explicaciones largas o de relleno.
2. Si la consulta implica una infracción, sanción o denuncia, debes estructurarla obligatoriamente así, salvo que tengas más conceptos para poner:
   - **Código:** [Código de la infracción si aparece, ej: AA01.05]
   - **Concepto:** [Descripción breve]
   - **Normas Infringidas:** [Artículos o normas aplicables]
   - **Cuantía:** [Importe si consta]
   - **Inmovilización:** [Indica si procede o no según la normativa]
3. Bástate exclusivamente en la información contenida en el texto.
4. Si encuentras algún texto con la explicación del concepto para denunciar, puedes preguntar si el usuario la quiere y entonces ponerla.
5. Si la información solicitada NO aparece en los textos proporcionados, responde únicamente: "No lo he visto".

Normativa Recuperada:
{contexto_filtrado}

Consulta del usuario:
{consulta_usuario}

Respuesta:
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

        except Exception as e:
            st.error(f"Error procesando la consulta: {e}")
