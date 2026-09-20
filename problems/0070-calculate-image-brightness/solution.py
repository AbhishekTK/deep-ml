
def calculate_brightness(img):
	# Write your code here
	if img is None or len(img) == 0:

		return -1
	s = 0
	n = len(img)
	rl = len(img[0])
	for r in img:
		for p in r:
			if p<0 or p>255:
				return -1
		if len(r)!=rl :


			return -1
		s += sum(r)
	# print(n)
	# print(s)
	return s/(n*n)