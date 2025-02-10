import cv2

def print_name(im, name):
	im = cv2.putText(im, name, (10, im.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
	return im

chicken = cv2.imread("img/chicken.jpg")
chicken = print_name(chicken, "John Doe | Jane Smith")
cv2.imwrite("chicken_named.jpg", chicken)

cv2.namedWindow("chicken")
cv2.imshow("chicken", chicken)
cv2.waitKey()
cv2.destroyAllWindows()
