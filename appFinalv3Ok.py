"""
Run a rest API exposing the yolov5s object detection mode
"""
import argparse
import io                    # Librerias para manejar carpetas del sistema operativo
from PIL import Image        # Transformacion de imagenes
import shutil                # Eliminacion de carpetas de sistema operativo
from shutil import rmtree
import os                    # Manejo del sistema operativo
import boto3                 # Para el servicio de predicción de edad
import requests, time              # Para controlar sistema operativo

import torch
from flask import Flask, jsonify, render_template # Lib para crear el servidor web
from flask_ngrok2 import run_with_ngrok # Lib para crear la URL publica
from flask import url_for
from flask import request    # Manejo de métodos de captura de APIs

from reportlab.lib.utils import Image

application = Flask(__name__)
# run_with_ngrok(application,auth_token='2Jpdf1TR33XB22x6UojejhuqxVu_5qQPXAy7RprfXSfgCPcmD') # Linea para indicar que se arrancara el servidor con Ngrok DBAExpert
run_with_ngrok(application,auth_token='2JptMb6IwVP0TlvTDfl77WG9pPl_4hAXcfFXxYqMEQ2RdN31e') # Linea para indicar que se arrancara el servidor con Ngrok Practicantes

@application.route("/send-image2/<path:url>", methods=['POST']) # Se asigna la direccion y se indica que admite el metodo POST
def predictUrl(url):
        # Captura los Datos recibidos y obtiene el dato que tiene la llave "image"
        # Lee el archivo
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img.save("./images/foto_descargada.jpg")

        results = model(img, size=640) # Pasa la imagen al modelo con un tamaño de imagen de 640px de ancho
        results.save() # Guarda la imagen con la deteccion en la carpeta run/detect/exp

        contenido = os.listdir('./runs/detect/exp') # Almacena el nombre de la imagen en contenido, posicion 0
        shutil.copy('./runs/detect/exp/'+contenido[0], './static/foto_detectada.jpg') # Copia la imagen a la carpeta static con el nombre "foto_detectada.jpg"
        rmtree('./runs/detect/exp') # Se elimina la carpeta runs con sus respectivas subcarpetas

        data = results.pandas().xyxy[0] # Se almacenan los parametros de detección

        # Generar la url del PDF generado
        urlsended = url_for('static', filename='Pdf_consulta_'+str(request.json["celular"])+'.pdf')
        prom=age()
        #global imperfeccionValue
        inperfeccionValue=""

        if(len(data) == 0):
            responder = {'nameURL': urlsended, 'Tipo Imperfección': "Rostro bien cuidado (No tienes imperfecciones)",
            'Predicción edad': 'Edad promedio ' + str(prom)}
            imperfeccionValue = "No se detectan imperfecciones"

        if(len(data) == 1):
            responder = {'nameURL': urlsended, 'Tipo Imperfección': str(data.values[0][6]),
            'Predicción edad': 'Edad promedio ' + str(prom)}
            imperfeccionValue = str(data.values[0][6])
            if(data.values[0][6] == "Acne"):
                imperfeccionValue = "Acné"

        if(len(data) == 2):
            responder = {'nameURL': urlsended, 'Tipo Imperfección': str(data.values[0][6])
            + ", " + str(data.values[1][6]),
            'Predicción edad': 'Edad promedio ' + str(prom)}
            if(data.values[0][6] == data.values[1][6]):
                imperfeccionValue = str(data.values[0][6])
            else:
                imperfeccionValue = str(data.values[0][6]) + ", " + str(data.values[1][6])

            if(data.values[0][6] == "Acne"):
                imperfeccionValue = "Acné"
            if(data.values[1][6] == "Acne"):
                imperfeccionValue = "Acné"

        varTipoPiel,combinacionTipo=comparacionesActivos()
        activo,linea,rec,rec2=principiosActivos(varTipoPiel,combinacionTipo)

        genPDFLocal(imperfeccionValue,prom,varTipoPiel,activo,linea,rec,rec2)
        # time.sleep(2)
        return jsonify(responder) # Envia el PDF generado con la imagen y valor de la detección en campo DetectionVal
# Método para cargar la imagen que se va a analizar
def age():
        # photo= './images/foto_descargada.jpg'
        photo= './static/foto_detectada.jpg'
        face_count=detect_faces(photo)
        return face_count
# Método para consumir servicio en AWS en el cual se realiza el promedio de la edad
def detect_faces(photo):
    # Conexión al servicio de AWS key
    client = boto3.client('rekognition',
                        aws_access_key_id="AKIA47P5BXX47DPG5JG5",
                        aws_secret_access_key="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",
                        region_name="us-east-1")
    # Usa el método de detección de rostro
    with open(photo, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])
    # Usa FaceDetails para estimar el rango de edad alto y bajo de la imagen y sacar un promedio
    for faceDetail in response['FaceDetails']:
        age_hight = faceDetail['AgeRange']['High']
        age_low = faceDetail['AgeRange']['Low']
        prom = (age_hight + age_low)/2
        prom = int(prom)
        if (prom < 15):
            prom = prom + 18
    return prom
# Método para la comparación de las variables capturadas por el bot para asignar el valor de tipo de piel
def comparacionesActivos():
    dataJSON = request.json
    p0 = dataJSON["pregunta_1"]
    p1 = dataJSON["pregunta_2"]
    p2 = dataJSON["pregunta_3"]
    p3 = dataJSON["pregunta_4"]
    p4 = dataJSON["pregunta_5"]

    varTipoPiel = ""
    combinacionTipo = ""

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "Si") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Seca-Sensible"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "Si") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Seca-Sensible"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "No") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Seca"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "No") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Seca"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta-Sensible"
        combinacionTipo = "TTO"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta-Sensible"
        combinacionTipo = "TTO"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta"
        combinacionTipo = "TTO"

    if((p0 == "Tirante") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta"
        combinacionTipo = "TTO"

    # if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "No") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "No") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa-Sensible"

    if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa-Sensible"

    if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa"
        combinacionTipo = "TOO"

    if((p0 == "Tirante") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa"
        combinacionTipo = "TOO"

    # if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "No") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Tirante") and (p3 == "No") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta-Sensible"
        combinacionTipo = "OTO"

    if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta-Sensible"
        combinacionTipo = "OTO"

    if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta"
        combinacionTipo = "OTO"

    if((p0 == "Oleosa") and (p1 == "Tirante") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Mixta"
        combinacionTipo = "OTO"

    # if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "Si") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "No") and (p4 == "Si")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    # if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Tirante") and (p3 == "No") and (p4 == "No")):
    #     varTipoPiel = "NO EXISTE LA COMBINACIÓN"

    if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa-Sensible"

    if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "Si") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa-Sensible"

    if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "Si") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa"
        combinacionTipo = "OOO"

    if((p0 == "Oleosa") and (p1 == "Oleosa") and (p2 == "Oleosa") and (p3 == "No") and ((p4 == "No") or (p4 == "N/A"))):
        varTipoPiel = "Piel Grasa"
        combinacionTipo = "OOO"

    return varTipoPiel, combinacionTipo
# Método para los activos dependiendo del valor que tenga asignada la variable tipo de piel
def principiosActivos(varTipoPiel,combinacionTipo):
    activo = ""
    linea = ""
    rec = ""
    rec2 = ""

    # if (varTipoPiel == "Piel Grasa"): #Combinacion: Tirante, Oleosa, Oleosa /// Maquillaje: NO
    #     activo = "Ácido hialuronico, Glicerina, Xilitol, Ácido agarico, Ácido salicilico, Enoxolona, Sulfato de zinc, Sulfato de cobre, Vitamina B6, Ramnosa, Bakuchiol. "
    #     linea = "Sensibio-Sebium"
    #     rec = "Se recomienda utilizar una rutina de limpieza dos veces al día tipo espuma, gel o loción, hidratación, tratamiento antiacné y fotoprotección."
    #     rec2 = "Oleosidad en todo el rostro, no solo en la zona T. Poros dilatados o visibles y pueden aparecer brotes de acné, espinillas y otras imperfecciones."

    if ((varTipoPiel == "Piel Grasa") and (combinacionTipo == "TOO")): #Combinacion: Tirante, Oleosa, Oleosa /// Maquillaje: N/A
        activo = "Ácido hialuronico, Glicerina, Xilitol, Ácido agarico, Ácido salicilico, Enoxolona, Sulfato de zinc, Sulfato de cobre, Vitamina B6, Gluconato de zinc. "
        linea = "Sensibio-Sebium"
        rec = "Se recomienda utilizar una rutina de limpieza dos veces al día tipo espuma, gel o loción, hidratación, tratamiento antiacné y fotoprotección."
        rec2 = "Oleosidad en todo el rostro, no solo en la zona T. Poros dilatados o visibles y pueden aparecer brotes de acné, espinillas y otras imperfecciones."

    if ((varTipoPiel == "Piel Grasa")  and (combinacionTipo == "OOO")): #Combinacion: Oleosa, Oleosa, Oleosa /// Maquillaje: NO & N/A
        activo = "Ácido hialuronico, Glicerina, Xilitol, Ácido agarico, Ácido salicilico, Enoxolona, Sulfato de zinc, Sulfato de cobre, Vitamina B6, Silica, Dióxido de titanium, Gingo biloba, Inflastop, Gluconato de zinc, Ácido glicolico, Glucosido de caprilo. "
        linea = "Sensibio-Sebium"
        rec = "Se recomienda utilizar una rutina de limpieza dos veces al día tipo espuma, gel o loción, hidratación, tratamiento antiacné y fotoprotección."
        rec2 = "Oleosidad en todo el rostro, no solo en la zona T. Poros dilatados o visibles y pueden aparecer brotes de acné, espinillas y otras imperfecciones."

    if (varTipoPiel == "Piel Grasa-Sensible"):
        activo = "Retinol, Vitamina C, B5, E, Niacinamida, Ácido hialurónico, Agua termal, Fotoprotección, Cafeína, Ceramidas. "
        linea = "Sensibio-Sebium"
        rec = "Se recomienda utilizar una rutina de limpieza para disminuir la oleosidad, calma la piel con un tónico, aplicar una hidratante específica para pieles grasas y sensibles, indispensable fotoproteger la piel."
        rec2 = "Oleosidad en todo el rostro, no solo en la zona T. Poros dilatados o visibles y pueden aparecer brotes de acné, espinillas, resequedad, molestia, rigidez, tirantez, picor, ardor y otras imperfecciones."

    if (varTipoPiel == "Piel Seca"):
        activo = "Ácido hialuronico, Vitamina E, Glucosido de coco, Glicerina, Canola, Xilitol, Extracto de semilla de manzana. "
        linea = "Sensibio-Hydrabio"
        rec = "Se recomienda mantener una rutina diaria de limpieza y mantener una hidratación durante la mañana y en la noche para proteger la piel de agresiones, indispensable fotoproteger la piel."
        rec2 = "Sensación de tirantez en todo el rostro, los poros son casi imperceptibles, además de agua carece de lípidos y dependiendo de la sequedad será escamosa."

    if (varTipoPiel == "Piel Seca-Sensible"): #Si N/A el maquillaje
        activo = "Ácido hialurónico, Vitamina E, Enoxolona, Alantoina, Glucosido de coco, Glicerina, Canola, Xilitol, Manitol, Ramnosa, Carnsina. "
        linea = "Sensibio-Hydrabio"
        rec = "Se recomienda mantener una rutina diaria de limpieza utilizando productos que protejan la piel de agresiones externas, evita frotar la piel porque son movimientos que puede irritar la piel. Es fundamental mantener una hidratación en la mañana y en la noche para mejorar el aspecto de la piel y aumentar el umbral de tolerancia de la piel, indispensable fotoproteger la piel."
        rec2 = "Sensación de tirantez en todo el rostro, los poros son casi imperceptibles, además de agua carece de lípidos, dependiendo de la sequedad será escamosa y pueden aparecer resequedad, molestia, rigidez, tirantez, picor, ardor y otras imperfecciones."

    # if (varTipoPiel == "Piel Seca-Sensible"): #Si el maquillaje le dura
    #     activo = "Ácido hialurónico, Vitamina E, Enoxolona, Alantoina, Glucosido de coco, Glicerina, Canola, Xilitol, Polifenoles de salvia roja. "
    #     linea = "Sensibio-Hydrabio"
    #     rec = "Se recomienda mantener una rutina diaria de limpieza utilizando productos que protejan la piel de agresiones externas, evita frotar la piel porque son movimientos que puede irritar la piel. Es fundamental mantener una hidratación en la mañana y en la noche para mejorar el aspecto de la piel y aumentar el umbral de tolerancia de la piel, indispensable fotoproteger la piel."
    #     rec2 = "Sensación de tirantez en todo el rostro, los poros son casi imperceptibles, además de agua carece de lípidos, dependiendo de la sequedad será escamosa y pueden aparecer resequedad, molestia, rigidez, tirantez, picor, ardor y otras imperfecciones."

    if ((varTipoPiel == "Piel Mixta")  and (combinacionTipo == "TTO")): #Combinacion: Tirante, Tirante, Oleosa //// Combinacion: Oleosa, Tirante, Oleosa. Maquillaje: N/A
        activo = "Ácido hialurónico, Glicerina, Xilitol, Ácido agarico, Ácido salicilico, Enoxolona, Sulfato de zinc, Sulfato de cobre. "
        linea = "Sensibio-Sebium"
        rec = "Se recomienda utilizar una rutina de limpieza dos veces al día tipo espuma, gel o loción, hidratación y fotoprotección."
        rec2 = "Presenta un aspecto brillante sobre todo en la llamada zona T, presenta pómulos con poros imperceptibles y secos."

    if ((varTipoPiel == "Piel Mixta") and (combinacionTipo == "OTO")): #Combinacion: Oleosa, Tirante, Oleosa. Maquillaje: NO
        activo = "Ácido hialurónico, Xilitol, Ácido agarico, Ácido salicilico, Enoxolona, Sulfato de zinc, Sulfato de cobre, Vitamnia B6, Silica, Dioxido de titanium, Gingo biloba. "
        linea = "Sensibio-Sebium"
        rec = "Se recomienda utilizar una rutina de limpieza dos veces al día tipo espuma, gel o loción, hidratación y fotoprotección."
        rec2 = "Presenta un aspecto brillante sobre todo en la llamada zona T, presenta pómulos con poros imperceptibles y secos."

    if ((varTipoPiel == "Piel Mixta-Sensible") and (combinacionTipo == "TTO")): #Combinacion: Tirante, Tirante, Oleosa
        activo = "Ácido hialurónico, Glicerina, Xilitol, Ácido agarico, Ácido salicilico, Enoxolona, Sulfato de zinc, Sulfato de cobre. "
        linea = "Sensibio-Sebium"
        rec = "Se recomienda utilizar productos de limpieza adecuados para tu tipo de piel en especial agua micelar o tónicos, adicionalmente productos que ayuden a minimizar los poros en la zona T y aplicar hidratante tipo textura serum, gel o emulsión, indispensable fotoproteger la piel."
        rec2 = "Presenta un aspecto brillante sobre todo en la llamada zona T, presenta pómulos con poros imperceptibles, secos y pueden aparecer resequedad, molestia, rigidez, tirantez, picor, ardor y otras imperfecciones."
    
    # if (varTipoPiel == "Piel Mixta-Sensible"): #Combinacion: Oleosa, Tirante, Oleosa /// Maquillaje: NO
    #     activo = "Ácido hialurónico, Glicerina, Xilitol, Ácido agarico, Ácido salicilico, Enoxolona, Sulfato de zinc, Sulfato de cobre, Vitamnia B6, Silica, Dióxido de titanium, Gingo biloba. "
    #     linea = "Sensibio-Sebium"
    #     rec = "Se recomienda utilizar productos de limpieza adecuados para tu tipo de piel en especial agua micelar o tónicos, adicionalmente productos que ayuden a minimizar los poros en la zona T y aplicar hidratante tipo textura serum, gel o emulsión, indispensable fotoproteger la piel."
    #     rec2 = "Presenta un aspecto brillante sobre todo en la llamada zona T, presenta pómulos con poros imperceptibles, secos y pueden aparecer resequedad, molestia, rigidez, tirantez, picor, ardor y otras imperfecciones."

    if ((varTipoPiel == "Piel Mixta-Sensible") and (combinacionTipo == "OTO")): #Combinacion: Oleosa, Tirante, Oleosa /// Maquillaje: N/A
        activo = "Ácido hialurónico, Glicerina, Xilitol, Ácido agarico, Ácido salicilico, Enoxolona, Sulfato de zinc, Sulfato de cobre, Perlita, Silica, Propil galato. "
        linea = "Sensibio-Sebium"
        rec = "Se recomienda utilizar productos de limpieza adecuados para tu tipo de piel en especial agua micelar o tónicos, adicionalmente productos que ayuden a minimizar los poros en la zona T y aplicar hidratante tipo textura serum, gel o emulsión, indispensable fotoproteger la piel."
        rec2 = "Presenta un aspecto brillante sobre todo en la llamada zona T, presenta pómulos con poros imperceptibles, secos y pueden aparecer resequedad, molestia, rigidez, tirantez, picor, ardor y otras imperfecciones."
    return activo,linea,rec,rec2

# Método para generar el PDF final de diagnóstico usando ReportLAB Python
def genPDFLocal(imperfeccionValue,prom,varTipoPiel,activo,linea,rec,rec2):
    dataJSON = request.json
    name_user = dataJSON["nombre_cliente"]
    experiencia_foto = dataJSON["experiencia_foto"]

    import datetime
    from reportlab.pdfgen import canvas # Librerias para la generación del PDF
    from reportlab.lib.utils import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import mm
    from reportlab.lib.utils import simpleSplit
    from reportlab.platypus import Paragraph
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle

    custom_size = (277.67*mm,490*mm)
    i = mm
    d = i/4
    w, h = custom_size
    c = canvas.Canvas("./static/Pdf_consulta_"+str(request.json["celular"])+".pdf",pagesize=custom_size)

    # Definir una función para dibujar un párrafo en una coordenada específica
    def draw_paragraph(canvas, x, y, width, text, style):
        p = Paragraph(text, style)
        p.wrapOn(canvas, width, 0)
        p.drawOn(canvas, x, y - p.height)

    c.setFont("Helvetica", 15)
    #Cambiar el Color del Fondo
    c.setFillColorRGB(1,1,1)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    if(experiencia_foto == "Si"):
        # Dimensiones Cambiaron definido como "custom_size = (294*mm,298*mm)" y en milimetros
        fotoia = ImageReader('./images/foto_descargada.jpg')
        c.drawImage(fotoia, 37*mm, 150*mm , width=200*mm ,  height=270*mm, preserveAspectRatio=False)

    if(experiencia_foto == "No"):
        fotoia = ImageReader('./imagesPDF/avatar.PNG')
        c.drawImage(fotoia, 40*mm, 190*mm , width=200*mm ,  height=200*mm, preserveAspectRatio=False)
        prom = ""

    bg = Image.open("./imagesPDF/fondo_v1.png")
    bg.save("./imagesPDF/fondo_tranparente.png")
    bg = ImageReader("./imagesPDF/fondo_tranparente.png")
    width, height = bg.getSize()
    c.drawImage(bg, x= 0, y=0, width=277.67*mm, height=490*mm, mask='auto')

    #_________________________________________________________________________
    c.setFont("Helvetica-Bold", 28)
    c.setFillColorRGB(0,0,0,1)

    def text_wrap(text, width):
        lines = []
        for line in simpleSplit(text, "Times-Roman", 8, width):
            lines.append(line)
        return lines

    text = varTipoPiel
    width = 160
    height = 10
    x = w/2
    y = 424.5 * mm

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15
    #___________________________________________________________
        #La edad de su piel es
    c.setFont("Helvetica-Bold", 34)
    c.setFillColorRGB(0,0,0,1)

    text = str(prom)
    width = 100
    height = 10
    x = 218 * mm
    y = 230 * mm

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15
    #___________________________________________________________
        #Nombre del cliente
    c.setFont("Helvetica-Bold", 20)
    c.setFillColorRGB(0,0,0,1)

    text = name_user
    width = 100
    height = 10
    x = 650
    y = 525

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15
    #___________________________________________________________
        #Fecha
    c.setFont("Helvetica-Bold", 20)
    c.setFillColorRGB(0,0,0,1)
    # Obtenemos la fecha y hora actual
    fecha_actual = datetime.datetime.now()
    # Convertimos la fecha a una cadena de texto en el formato deseado
    # fecha_actual_str = fecha_actual.strftime("%d/%m/%Y %H:%M:%S")

    # Definimos una lista de nombres de meses
    nombres_meses = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]

    # Obtenemos el día, el mes y el año como enteros
    dia = fecha_actual.day
    mes = fecha_actual.month
    anio = fecha_actual.year

    # Creamos la cadena de texto con el formato deseado
    fecha_str = f"{dia} de {nombres_meses[mes - 1]} del {anio}"

    text = fecha_str
    width = 100
    height = 10
    x = 170
    y = 525

    for line in text_wrap(text, width):
        c.drawCentredString(x, y, line)
        y -= 15
    #___________________________________________________________
    # Parrafo 1
    # estilo_normal = getSampleStyleSheet()["Normal"]
    estilo_personalizado = ParagraphStyle(
        "mi_estilo",
        fontSize=20,
        fontName="Helvetica",
        leading=18,
        alignment=TA_JUSTIFY
    )

    estilo_negrita = ParagraphStyle(
        "negrita",
        parent=estilo_personalizado,  # Heredar del estilo personalizado
        fontName="Helvetica-Bold",
        textColor="purple"
    )

    text = rec2
    draw_paragraph(c, 75, 445, 630, text, estilo_personalizado)
    #___________________________________________________________
    # Parrafo 2
    
    text = activo
    draw_paragraph(c, 75, 283, 630, text, estilo_personalizado)

    text = "Línea recomendada: " + linea
    draw_paragraph(c, 75, 223, 630, text, estilo_negrita)
    #___________________________________________________________
    # Parrafo 3
    text = rec
    draw_paragraph(c, 75, 145, 630, text, estilo_personalizado)
    #___________________________________________________________
    c.showPage()
    c.save()

@application.route('/none') # Ruta para prueba de funcionamiento,  Solo muestra el memsaje de hola en el navegador
def none():
    return render_template('index.html') # se debe llamar con GET

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # Carga el detector con el modelo COCO
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) # Carga el detector con el modelo Sonrisas
    model.conf = 0.7 # Indica el nivel de confianza minimo en la detección
    model.eval()

    #application.run(host="0.0.0.0", port=4000, debug=True)  # Inicia en servidor Local
    application.run() # inicia en Servidor Remoto,  Tener en cuenta que en cada inicio de servidor esta direccion cambia
    # debido a que se esta usando una libreria gratuita de tunelamiento.
