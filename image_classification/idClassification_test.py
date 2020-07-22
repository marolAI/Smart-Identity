from idClassification import id_not_id
import cv2
import imutils


# load the image
image = cv2.imread("examples/bank.jpeg")
orig = image.copy()

label, proba = id_not_id(image, model="id_not_id.model")

label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

print(label, proba)
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
