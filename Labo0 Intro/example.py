import cv2
import os

def print_name(im, name):
	im = cv2.putText(im, name, (10, im.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
	return im

def main():
	file_path = "img/chicken.jpg"
	if not os.path.exists(file_path):
		message = f"Error: Could not open file {file_path}"
		print((message))
		exit(-1)
	chicken = cv2.imread(file_path)
	chicken = print_name(chicken, "John Doe | Jane Smith")
	cv2.imwrite("chicken_named.jpg", chicken)

	cv2.namedWindow("chicken")
	cv2.imshow("chicken", chicken)
	cv2.waitKey()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	main()