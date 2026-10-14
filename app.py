if consulta_usuario:
    with st.spinner("Buscando en la normativa y redactando respuesta..."):
        try:
            # ... (código de consulta a ChromaDB) ...

            contexto = "\n\n---\n\n".join(documentos_recuperados)

            # Alinear la línea exactamente sin espacios extra al inicio:
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

            # Respuesta con temperatura 0.2
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
