import streamlit as st
import glob
import os
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

st.set_page_config(page_title="Megaleyes - Consulta Legal", layout="wide")
st.title("⚖️ Megaleyes: Asistente Legal")

@st.cache_resource
def cargar_leyes():
    # Lee todos los archivos .txt y .md de la carpeta 'leyes'
    archivos = glob.glob("leyes/*.txt") + glob.glob("leyes/*.md")
    textos = []
    
    for ruta in archivos:
        with open(ruta, "r", encoding="utf-8") as f:
            nombre_archivo = os.path.basename(ruta)
            textos.append(f"--- INICIO ARCHIVO: {nombre_archivo} ---\n{f.read()}\n--- FIN ARCHIVO ---")
            
    # Une todos los archivos en una sola gran masa de conocimiento
    return "\n\n".join(textos), len(archivos)

@st.cache_resource
def inicializar_gemini():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=HttpOptions(api_version="v1")
    )

try:
    contexto_completo, num_archivos = cargar_leyes()
    cliente_gemini = inicializar_gemini()
    st.sidebar.success(f" Base de conocimiento cargada: {num_archivos} archivos leídos.")
except Exception as e:
    st.error(f"Error cargando las leyes o cliente: {e}")
    st.stop()

consulta_usuario = st.text_input("Haz tu consulta legal:")

if consulta_usuario:
    with st.spinner("Analizando la normativa completa y redactando respuesta..."):
        try:
            prompt = f"""
Eres un asistente legal de máxima precisión. Tienes a continuación la normativa legal completa. Tu objetivo es analizarla toda para responder a la consulta del usuario relacionando conceptos.

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

Normativa Legal Completa:
{contexto_completo}

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
