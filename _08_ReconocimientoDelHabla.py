# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

pip install SpeechRecognition pyaudio

import speech_recognition as sr

# Inicializar el reconocedor de voz
reconocedor = sr.Recognizer()

# Inicializar el micrófono
microfono = sr.Microphone()

# Configurar el micrófono (puedes ajustar parámetros según tus necesidades)
with microfono as source:
    reconocedor.adjust_for_ambient_noise(source)

# Escuchar y reconocer el habla
try:
    print("Habla algo...")
    with microfono as source:
        audio = reconocedor.listen(source)
    texto = reconocedor.recognize_google(audio, language="es-ES")  # Reconocimiento con Google

    print("Texto reconocido:", texto)

except sr.UnknownValueError:
    print("No se pudo entender el habla.")
except sr.RequestError as e:
    print(f"Error en la solicitud: {e}")
