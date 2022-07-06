from graphics import *
import time
import math
import numpy as np

def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

def clear(win):
	for item in win.items[:]:
		item.undraw()

class Point3D():
	x = 0
	y = 0
	z = 0
	
	def __init__(self, X = 0, Y = 0, Z = 0):
		self.x = X
		self.y = Y
		self.z = Z
		
	def set_coordinates(self, X, Y, Z):
		self.x = X
		self.y = Y
		self.z = Z
	
	def set_x(self, X):
		self.x = X
	
	def set_y(self, Y):
		self.y = Y
	
	def set_z(self, z):
		self.z = Z
	
	def get_vector(self):
		return np.array([self.x, self.y, self.z])
	
	def get_vector4(self):
		return np.array([self.x, self.y, self.z, 1])
	
	def convert_to_2D(self, point, x, y, p, win):
		point_2D = func_3d_to_2d(self.get_vector4(), point, x, y, p, win)
		return Point(point_2D[0], point_2D[1])


def world_to_spectator(vector, point):
	T = np.eye(4)
	T[3][0] = -point.x
	T[3][1] = -point.y
	T[3][2] = -point.z
	return vector @ T

def cs_x_invert_y_up(vector):
	SR = np.zeros((4, 4))
	SR[0][0] = -1
	SR[1][2] = -1
	SR[2][1] = 1
	SR[3][3] = 1
	return vector @ SR

def cs_z_to_cs_start(vector, point):
	R1 = np.eye(4)
	R2 = np.eye(4)
	d = (point.x ** 2 + point.y ** 2) ** 0.5
	s = (point.x ** 2 + point.y ** 2 + point.z ** 2) ** 0.5
	if (d != 0):
		R1[0][0] = point.y / d
		R1[0][2] = point.x / d
		R1[2][0] = -point.x / d
		R1[2][2] = point.y / d
	if (s != 0):
		R2[1][1] = d / s
		R2[1][2] = -point.z / s
		R2[2][1] = point.z / s
		R2[2][2] = d / s
	return vector @ R1 @ R2

def paralell(vector):
	return np.array([vector[0], vector[1]])

def perspective(vector, point):
	s = (point.x ** 2 + point.y ** 2 + point.z ** 2) ** 0.5
	return np.array([vector[0] * s / vector[2], vector[1] * s / vector[2]])

def screen_cs(vector, x, y, p, win):
	return np.array([vector[0] * x / p + win.width / 2, win.height / 2 - vector[1] * y / p])

def func_3d_to_2d(vector, point, x, y, p, win):
	return screen_cs(paralell(cs_z_to_cs_start(cs_x_invert_y_up(world_to_spectator(vector, point)), point)), x, y, p, win)

def make_convert_matrix(point):
	return cs_z_to_cs_start(cs_x_invert_y_up(world_to_spectator(np.eye(4), point)), point)

def func_3d_to_2d_with_matrix(vector, matrix, x, y, p, win):
	return screen_cs(paralell(vector @ matrix), x, y, p, win)

def func_3d_to_pks(vector, point):
	return paralell(cs_z_to_cs_start(cs_x_invert_y_up(world_to_spectator(vector, point)), point))

def line_convert(point1, point2, point_view, x, y, p ,win):
	return np.array([func_3d_to_2d(point1.get_vector4(), point_view, x, y, p, win), func_3d_to_2d(point2.get_vector4(), point_view, x, y, p, win)])

def line_convert_p_arr(line, point_view, x, y, p ,win):
	return np.array([func_3d_to_2d(line[0].get_vector4(), point_view, x, y, p, win), func_3d_to_2d(line[1].get_vector4(), point_view, x, y, p, win)])

def line_draw(line, win, color = 'black', width = 1):
	line = Line(Point(line[0][0], line[0][1]), Point(line[1][0], line[1][1]))
	line.setOutline(color)
	line.setWidth(width)
	line.draw(win)

def dotted_line_draw(line, win, length = 20, color = 'black'):
	dx = line[1][0] - line[0][0]
	dy = line[1][1] - line[0][1]
	
	if (dx == 0 and dy == 0):
		return
	
	prev_x = line[0][0]
	prev_y = line[0][1]
	
	if (dx != 0):
		c = ((length ** 2) / ((dy / dx) ** 2 + 1)) ** 0.5
		a = dy / dx
		b = (line[0][1] * line[1][0] - line[1][1] * line[0][0]) / dx
	else:
		c = 0
		a = 0
		b = 0
	
	for i in range(round(math.sqrt((dx ** 2 + dy ** 2)) / (length * 2))):
		x1 = prev_x
		y1 = prev_y
		
		x2 = c * sign(dx) + x1
		y2 = a * x2 + b
		
		prev_x = c * sign(dx) + x2
		prev_y = a * prev_x + b
		
		if (dx == 0):
			x2 = x1
			y2 = y1 + length * sign(dy)
			prev_x = x1
			prev_y = y2 + length * sign(dy)
		if (dy == 0):
			y2 = y1
			x2 = x1 + length * sign(dx)
			prev_y = y1
			prev_x = x2 + length * sign(dx)
		
		line_temp = Line(Point(x1, y1), Point(x2, y2))
		line_temp.setOutline(color)
		line_temp.draw(win)


def function(x, z):
	return x**2-z**2#z ** 3 + x * z

def window(xmin, xmax, zmin, zmax, dx, dz, uy, ux, point = Point3D(0, 0, -100)):
	X = np.zeros((4, 2))
	point_temp = Point3D(xmin, 0, zmin)
	X[0] = func_3d_to_pks(point_temp.get_vector4(), point)
	point_temp = Point3D(xmax, 0, zmin)
	X[1] = func_3d_to_pks(point_temp.get_vector4(), point)
	point_temp = Point3D(xmin, 0, zmax)
	X[2] = func_3d_to_pks(point_temp.get_vector4(), point)
	point_temp = Point3D(xmax, 0, zmax)
	X[3] = func_3d_to_pks(point_temp.get_vector4(), point)
	result = np.zeros(2)
	result[0] = abs(X[0][0])
	for i in range(1, 4):
		if (result[0] < abs(X[i][0])):
			result[0] = abs(X[i][0])
	result[1] = 0
	count_z = int((zmax - zmin) / dz)
	count_x = int((xmax - xmin) / dx)
	x = xmin
	z = zmin
	sin_x = math.sin(ux)
	cos_x = math.cos(ux)
	sin_y = math.sin(uy)
	cos_y = math.cos(uy)
	for i in range(count_z):
		for j in range(count_x):
			y = function(x, z)
			x_temp = x
			coordinates = np.array([x, y, z, 1])
			if (uy != 0):
				coordinates = coordinates @ np.array([[cos_y, 0, sin_y, 0], [0, 1, 0, 0], [-sin_y, 0, cos_y, 0], [0, 0, 0, 1]])
			if (ux != 0):
				coordinates = coordinates @ np.array([[1, 0, 0, 0], [0, cos_x, -sin_x, 0], [0, sin_x, cos_x, 0], [0, 0, 0, 1]])
			temp = func_3d_to_pks(coordinates, point)
			if (result[1] < abs(temp[1])):
				result[1] = abs(temp[1])
			x += dx
		z += dz
	return result

def correction(prev, t, up, down):
	if (prev[0] == t[0]):
		up[prev[0]] = max(prev[1], t[1], up[prev[0]])
		down[prev[0]] = min(prev[1], t[1], down[prev[0]])
	m = (t[1] - prev[1]) / (t[0] - prev[0])
	x = prev[0]
	y = prev[1]
	while (x <= t[0]):
		up[x] = max(y, up[x])
		down[x] = min(y, down[x])
		x += 1
		y += m

def insert_2(prev, point_draw, up, down, new_up, new_down, win):
	if ((point_draw[1] > up[point_draw[0]]) or (point_draw[1] < down[point_draw[0]])):
		Point(point_draw[0], point_draw[1]).draw(win)
		if (point_draw[1] > up[point_draw[0]]):
			new_up[point_draw[0]] = point_draw[1]
		if (point_draw[1] < down[point_draw[0]]):
			new_down[point_draw[0]] = point_draw[1]
	if ((prev[0] != -1) and ((prev[1] > up[prev[0]]) or (prev[1] < down[prev[0]]))):
		if (prev[1] > up[prev[0]]):
			new_up[prev[0]] = prev[1]
		if (prev[1] < down[prev[0]]):
			new_down[prev[0]] = prev[1]

def intersection(prev, t, up, down, win):
	x0 = 0
	y0 = 0
	if (prev[0] == t[0]):
		x0 = prev[0]
		y0 = up[prev[0]]
		return [x0, y0]
	else:
		m = (t[1] - prev[1]) / (t[0] - prev[0])
		sp = 0
		if (prev[1] > up[prev[0]]):
			sp = 1
		elif (prev[1] < up[prev[0]]):
			sp = -1
	x = prev[0]
	y = prev[1]
	s = sp
	while (s == sp):
		s = 0
		if (y > up[x]):
			s = 1
		elif (y < up[x]):
			s = -1
		x += 1
		y += m
		if (x == t[0]):
			return [t[0], t[1]]
	xl = x - 1
	yl = y - m
	xr = x + 1 * 0
	yr = y + m * 0
	if (abs(yl - up[xl]) <= abs(yr - up[xr])):
		x0 = xl
		y0 = yl
	else:
		x0 = xr
		y0 = yr
	if (abs(m) < abs(up[xr] - up[xl])):
		y0 = up[x0]
	return [x0, y0]

def insert_3(prev, point_draw, up, down, new_up, new_down, win):
	if (point_draw[1] > up[point_draw[0]]):
		new_up[point_draw[0]] = point_draw[1]
		point_draw[2] += 1
	if (point_draw[1] < down[point_draw[0]]):
		new_down[point_draw[0]] = point_draw[1]
		point_draw[2] += 2
	if (prev[0] == -1):
		return
	
	if ((point_draw[2] != 0 and prev[2] != 0) and (point_draw[2] == prev[2])):
		Line(Point(prev[0], prev[1]), Point(point_draw[0], point_draw[1])).draw(win)
	else:
		s = [prev[0], prev[1]]
		q = [-1, -1]
		if (prev[2] % 3 != 0):
			s = intersection(prev, point_draw, up, down, win)
			if (s[0] != -1):
				Line(Point(prev[0], prev[1]), Point(s[0], s[1])).draw(win)
		if (point_draw[2] % 3 != 0):
			q = intersection(s, point_draw, up, down, win)
			if (q[0] != -1):
				Line(Point(q[0], q[1]), Point(point_draw[0], point_draw[1])).draw(win)

def func_draw(xmin, xmax, zmin, zmax, dx, dz, uy, ux, xe, ye, p, point, win):
	#pkxy = window(xmin, xmax, zmin, zmax, dx, dz, uy, ux, point) # задать матрицу точек y
	pkxy = [xe, ye]
	count_z = int((zmax - zmin) / dz)
	count_x = int((xmax - xmin) / dx)
	up = np.repeat(int(0), win.width)
	down = np.repeat(win.height, win.width)
	new_up = np.copy(up)
	new_down = np.copy(down)
	x = xmax
	z = zmin
	sin_x = math.sin(ux)
	cos_x = math.cos(ux)
	sin_y = math.sin(uy)
	cos_y = math.cos(uy)
	convert_matrix = make_convert_matrix(point)
	for i in range(count_z):
		x = xmax
		prev = [-1, -1, 0]
		for j in range(count_x):
			y = function(x, z)
			coordinates = np.array([x, y, z, 1])
			if (uy != 0):
				coordinates = coordinates @ np.array([[cos_y, 0, sin_y, 0], [0, 1, 0, 0], [-sin_y, 0, cos_y, 0], [0, 0, 0, 1]])
			if (ux != 0):
				coordinates = coordinates @ np.array([[1, 0, 0, 0], [0, cos_x, -sin_x, 0], [0, sin_x, cos_x, 0], [0, 0, 0, 1]])
			point_draw = func_3d_to_2d_with_matrix(coordinates, convert_matrix, pkxy[0], pkxy[1], p, win)
			point_draw = [int(round(point_draw[0])), int(round(point_draw[1])), 0]
			insert_3(prev, point_draw, up, down, new_up, new_down, win)
			prev = point_draw
			x -= dx
		up = np.copy(new_up)
		down = np.copy(new_down)
		z += dz






def main():
	win = GraphWin("Surfaces (lab4)", 700, 600, autoflush=False)
	
	point = Point3D(0, 0, -300)
	x = 350
	y = 300
	p = 20
	
	xmin = -5
	xmax = 5
	zmin = -2
	zmax = 2
	dx = 0.1
	dz = 0.1
	func_draw(xmin, xmax, zmin, zmax, dx, dz, 0, 0, x, y, p, point, win)
	win.update()
	
	time.sleep(1)
	
	frames_coefficient = 1
	angle = math.pi / (180 * frames_coefficient)
	
	for i in range(30 * frames_coefficient):
		clear(win)
		func_draw(xmin, xmax, zmin, zmax, dx, dz, 0, 5 * (i + 1) * angle, x, y, p, point, win)
		win.update()
		time.sleep(0.1)
	print("Done!")
	
	
	win.getMouse()
	clear(win)
	win.close()

if __name__ == '__main__':
	main()
