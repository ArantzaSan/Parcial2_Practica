import speech_recognition as sr

def reconocer_voz():
    """
    Utiliza la biblioteca SpeechRecognition para capturar y reconocer el habla desde el micrófono.

    Returns:
        str: El texto reconocido, o None si no se pudo entender el audio.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Di algo...")
        r.adjust_for_ambient_noise(source)  # Opcional: calibrar el ruido ambiental
        try:
            audio = r.listen(source)
            print("Procesando...")
            texto = r.recognize_google(audio, language='es-MX')  # Utiliza el reconocimiento de Google en español mexicano
            print(f"Has dicho: {texto}")
            return texto
        except sr.UnknownValueError:
            print("No se pudo entender el audio.")
            return None
        except sr.RequestError as e:
            print(f"Error al solicitar resultados del servicio de reconocimiento; {e}")
            return None
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")
            return None

if __name__ == "__main__":
    texto_reconocido = reconocer_voz()
    if texto_reconocido:
        print(f"Texto reconocido en la variable: {texto_reconocido}")
        # Aquí puedes realizar acciones con el texto reconocido
