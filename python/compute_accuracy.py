import numpy as np
import math


def compute_accuracy(predictions_48, imdb, W_48):
	i = np.argmax(np.squeeze(predictions_48))
	idx = W_48[i,3]
	predicted_label = W_48[i,:]
	true_label = imdb[idx].pyramid[0].label

	tlabelwidth = true_label[2]
	txlabel_left = true_label[0]-int(tlabelwidth/2)
	txlabel_right = true_label[0]+int(tlabelwidth/2)
	tylabel_upper = true_label[1]-int(tlabelwidth/2)
	tylabel_lower = true_label[1]+int(tlabelwidth/2)


	plabelwidth = predicted_label[2]
	pxlabel_left = predicted_label[0]-int(plabelwidth/2)
	pxlabel_right = predicted_label[0]+int(plabelwidth/2)
	pylabel_upper = predicted_label[1]-int(plabelwidth/2)
	pylabel_lower = predicted_label[1]+int(plabelwidth/2)

	margin = 2/math.pow(tlabelwidth,2)
       	accx = 1- margin*(math.pow(pxlabel_left-txlabel_left,2)+ math.pow(pxlabel_right-txlabel_right,2))
	accy = 1- margin*(math.pow(pylabel_upper-tylabel_upper,2)+ math.pow(pylabel_lower-tylabel_lower,2))
	accx = max(accx, 0.0)
	accy = max(accy, 0.0)

	#errorx = math.pow(pxlabel_left-txlabel_left,2)+ math.pow(pxlabel_right-txlabel_right,2)
	#errory = math.pow(pylabel_upper-tylabel_upper,2)+ math.pow(pylabel_lower-tylabel_lower,2)
	acc = min(accx, accy)
	return(acc)


def compute_accuracy_dataset(maxlabels, imdb, W_48):
	acc = []
	for k in range(maxlabels.shape[0]):
		i = int(maxlabels[k][1])
		idx = W_48[i,3]
		predicted_label = W_48[i,:]
		true_label = imdb[idx].pyramid[0].label
	
		tlabelwidth = true_label[2]
		txlabel_left = true_label[0]-int(tlabelwidth/2)
		txlabel_right = true_label[0]+int(tlabelwidth/2)
		tylabel_upper = true_label[1]-int(tlabelwidth/2)
		tylabel_lower = true_label[1]+int(tlabelwidth/2)


		plabelwidth = predicted_label[2]
		pxlabel_left = predicted_label[0]-int(plabelwidth/2)
		pxlabel_right = predicted_label[0]+int(plabelwidth/2)
		pylabel_upper = predicted_label[1]-int(plabelwidth/2)
		pylabel_lower = predicted_label[1]+int(plabelwidth/2)

		margin = 2/math.pow(tlabelwidth,2)
	       	accx = 1- margin*(math.pow(pxlabel_left-txlabel_left,2)+ math.pow(pxlabel_right-txlabel_right,2))
		accy = 1- margin*(math.pow(pylabel_upper-tylabel_upper,2)+ math.pow(pylabel_lower-tylabel_lower,2))
		accx = max(accx, 0.0)
		accy = max(accy, 0.0)
	
		#errorx = math.pow(pxlabel_left-txlabel_left,2)+ math.pow(pxlabel_right-txlabel_right,2)
		#errory = math.pow(pylabel_upper-tylabel_upper,2)+ math.pow(pylabel_lower-tylabel_lower,2)
		acc.append(min(accx, accy))
	return(acc)
