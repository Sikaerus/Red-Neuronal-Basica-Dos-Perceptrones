#Sikaerus, GNU: México
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

def perceptron(x1,x2,w1,w2,b,F,s):
	producto_escalar = (x1*w1) + (x2*w2) + b
	if F == 'E':
		y = funcion_activacion_escalon(producto_escalar)
	elif F == 'S':
		print("ERROR No Sigmoid")

	X = [x1, x2]
	W = [w1, w2]

	n_B, n_w1, n_w2, e = reajustes_y_calculos_error_bias_pesos(X,W,y,b,s)#Obtiene el nuevo_bias, el nuevo_pesoW1 y el nuevo_pesoW2

	return n_B, n_w1, n_w2, y, e
	pass

def funcion_activacion_escalon(x):
	y = 0
	if x > 0.5:
		y = 1
	elif x <= 0.5:
		y = 0
	return y
	pass

def reajustes_y_calculos_error_bias_pesos(X,W,salida_obtenida, bias_actual, salida_esperada):

	α = 0.2#Factor de aprendizaje, alfa, alpha α

	error = salida_esperada - salida_obtenida#Calculo de error
	nuevo_bias = bias_actual + salida_esperada#Actualización de bias

	Δw1 = α*error*X[0]#Delta Δ reprecenta un cambio significativo en una variable
	nuevo_pesoW1 = W[0] + Δw1#Calculo de nuevos pesos, peso 1

	Δw2 = α*error*X[1]#Delta Δ reprecenta un cambio significativo en una variable
	nuevo_pesoW2 = W[1] + Δw2#Calculo de nuevos pesos, peso 2

	return nuevo_bias, nuevo_pesoW1, nuevo_pesoW2, error
	pass

def main():

	#Para un buen entrenamiento debe de existir un equilibrio entre EPOCAS e INSTANCIAS
	#Ya que pocas INSTANCIAS requieren mas EPOCAS sin paro de error = 0
	#Pero cuendo se tienen suficientes INSTANCIAS para entrenar el fin de entrenamiento
	#puede ser cuando el error = 0, de esta forma se evita llegar a las n EPOCAS
	#Si se tienen pocas instancias sin error de paro y con pocas EPOCAS será dificil hacer un buen entrenamiento
	INSTANCIAS = 20#Las INSTANCIAS reprecentan el número fijo de ejemplos de entrenamiento
	EPOCAS = 100

	FUNCION = 'E'
	contador = 0
	X1, X2, S1, S2, W = generar_datos(INSTANCIAS)

	for i in range(0, INSTANCIAS):
		bias1 = 0.1 #El bias o parcialidad nos ayuda a determinar correctamente la linea que separa las clases
		bias2 = 0.1
		while contador < EPOCAS:
			bias1, W[0], W[2], y1, e1 = perceptron(X1[i],X2[i],W[0],W[2],bias1,FUNCION,S1[i])
			print("Instancia ", i, "{ bias: ",bias1, " pesos(",W[0],W[2],")"," Salida Obtenida: ",y1," Salida Esperada: ",S1[i],"} Epoca:", contador)
			bias2, W[1], W[3], y2, e2 = perceptron(X1[i],X2[i],W[1],W[3],bias2,FUNCION,S2[i])
			print("Instancia ", i, "{ bias: ",bias2, " pesos(",W[1],W[3],")"," Salida Obtenida: ",y2," Salida Esperada: ",S2[i],"} Epoca:", contador)
			if e1 == 0 and e2 == 0:
				print("Error 1: ",e1, " Error 2: ",e2)
				break
			contador += 1
		print("---> Instancia ", i, "{ bias: ",bias1, " pesos(",W[0],W[2],")"," Salida Obtenida: ",y1," Salida Esperada: ",S1[i],"} Epoca:", contador)
		print("---> Instancia ", i, "{ bias: ",bias2, " pesos(",W[1],W[3],")"," Salida Obtenida: ",y2," Salida Esperada: ",S2[i],"} Epoca:", contador)
		contador = 0

	graficar(X1, X2, S1, S2)
	pass

def generar_datos(INSTANCIAS):#Recibe un número de INSTANCIAS/ejemplos que ha de crear

	W = np.zeros((4, 1))
	for i in range(0, len(W)):
		W[i] = random.uniform(-10, 10)#Vector de pesos aleatorios entre -10 y 10 con 4 elementos
	print(W, "Pesos random -10,10")#Entre MAYOR sea el RANGO de pesos (en este caso de [-10, 10]) MAS EPOCAS serán necesarias PARA CONVERGER para las primeras INSTANCIAS

	X1 = np.random.rand(INSTANCIAS, 1)#Vector de entrada aleatorios entre 0 y 1 con n INSTANCIAS
	X2 = np.random.rand(INSTANCIAS, 1)#Vector de entrada aleatorios entre 0 y 1 con n INSTANCIAS

	S1 = np.zeros(shape=(INSTANCIAS,1)); S2 = np.zeros(shape=(INSTANCIAS,1))#Genera un 2 vectores de n INSTANCIAS en ceros
	for i in range(0, INSTANCIAS):
		S1[i] = random.choice([0, 1])#Vector de salidas espaeradas S1, la salida obtenida es y_{1}
		if S1[i] == 0:
			S2[i] = 1;#Vector de salidas espaeradas S2, la salida obtenida es y_{1}.
		elif S1[i] != 0:
			S2[i] = 0;#Vector de salidas espaeradas S2, la salida obtenida es y_{1}.


	"""#DATOS FIJOS DE PRUEBA, VISTOS EN EL DOCUMENTO
	W = np.array([[3.80302985 ],[-2.89200548],[-6.59526659],[0.74931817 ]])
	print(W, "Pesos fijos de prueba -10, 10")
	#DATOS FIJOS DE PRUEBA, VISTOS EN EL DOCUMENTO
	X1 = np.array([[0.95662936],[0.87494675],[0.83615427],[0.83896432],[0.55903885],[0.90487343],[0.37507353],[0.16713584],[0.06229402],[0.19442781],[0.24181593],[0.08634414],[0.44564855],[0.0043262],[0.68332039],[0.53396922],[0.253226],[0.11206421],[0.3899378],[0.41176232]])
	X2 = np.array([[0.34734398],[0.73313805],[0.21178763],[0.77615795],[0.14243021],[0.25602497],[0.9620205],[0.0997963],[0.14997918],[0.7599588],[0.95950415],[0.17435251],[0.16966406],[0.64745794],[0.46816593],[0.32233097],[0.32531514],[0.85853743],[0.41040715],[0.34208895]])
	#DATOS FIJOS DE PRUEBA, VISTOS EN EL DOCUMENTO
	S1 = np.array([[0],[1],[1],[0],[0],[1],[0],[0],[0],[0],[0],[1],[1],[1],[0],[0],[0],[0],[0],[0]])
	S2 = np.array([[1],[0],[0],[1],[1],[0],[1],[1],[1],[1],[1],[0],[0],[0],[1],[1],[1],[1],[1],[1]])"""


	conjunto_datos = np.concatenate((X1, X2, S1, S2), axis=1)#Conjunto de datos
	print("     X_{1}     X_{2}     S_{1}     S_{2}")
	print(conjunto_datos, "Conjunto de datos de entrenamiento")
	return X1,X2,S1,S2,W
	pass


def graficar(X1, X2, S1, S2):

	plt.figure(figsize=(5,5))

	for s in range(0, len(S1)):
		if S1[s] == 0:
			punto_rojo, = plt.plot(X1[s], X2[s], color='red', marker='o', markeredgewidth=0.1, markersize=6)
		else:
			estrella_azul, = plt.plot(X1[s], X2[s], color='blue', marker='*', markeredgewidth=0.1, markersize=7)

	plt.xlabel("Valores en X1")
	plt.ylabel("Valores en X2")
	plt.grid(True)
	plt.title("Nube de puntos del conjunto de datos")
	#plt.legend(loc="upper right")
	plt.legend([punto_rojo, estrella_azul], ["C1(S1:0,S2:1)", "C2(S1:1,S2:0)"])
	plt.show()
	plt.ion()
	pass

if __name__ == '__main__':
	main()
