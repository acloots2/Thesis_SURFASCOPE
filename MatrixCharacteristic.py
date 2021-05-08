import numpy as np

def MatCharac(Mat, filename):
    MatCharac = open(filename, mode='w')

    MatCharac.write("The files contains the information for the matrix " + filename)
    MatCharac.write("\n\n")
    MatCharac.write("\n\nThe matrix has a shape " + str(Mat.shape))
    MatCharac.write("\n\nThe matrix has a size " + str(Mat.size))
    MatCharac.write("\n\nchi^0(0, 0) = " +str(Mat[0, 0, 0, 0, 0, 0]))
    MatCharac.write("\n\nThe max abs real value of chi^0 is " + str(np.amax(np.abs(np.real(Mat)))))
    MatCharac.write("\n\nThe max abs imag value of chi^0 is " + str(np.amax(np.abs(np.imag(Mat)))))
    sum1 = np.sum(Mat)

    sum2 = np.sum(np.power(np.real(Mat), 2) + np.power(np.imag(Mat), 2))

    MatCharac.write("\n\nint_v1 int_v2 chi^0 dv1 dv2 = " + str(sum1))
    MatCharac.write("\n\nint_v1 int_v2 |chi^0|^2 dv1 dv2 = " + str(sum2))

    MatCharac.close()


def MatCharacRec(Mat, filename):
    MatCharac = open(filename, mode='w')

    MatCharac.write("The files contains the information for the matrix " + filename)
    MatCharac.write("\n\n")
    MatCharac.write("\n\nThe matrix has a shape " + str(Mat.shape))
    MatCharac.write("\n\nThe matrix has a size " + str(Mat.size))
    MatCharac.write("\n\nchi^0(0, 0) = " +str(Mat[0, 0, 0]))
    MatCharac.write("\n\nThe max abs real value of chi^0 is " + str(np.amax(np.abs(np.real(Mat)))))
    MatCharac.write("\n\nThe max abs imag value of chi^0 is " + str(np.amax(np.abs(np.imag(Mat)))))
    sum1 = np.sum(Mat)

    sum2 = np.sum(np.power(np.real(Mat), 2) + np.power(np.imag(Mat), 2))

    MatCharac.write("\n\nint_v1 int_v2 chi^0 dv1 dv2 = " + str(sum1))
    MatCharac.write("\n\nint_v1 int_v2 |chi^0|^2 dv1 dv2 = " + str(sum2))

    MatCharac.close()

def MatCharacRec2D(Mat, filename):
    MatCharac = open(filename, mode='w')

    MatCharac.write("The files contains the information for the matrix " + filename)
    MatCharac.write("\n\n")
    MatCharac.write("\n\nThe matrix has a shape " + str(Mat.shape))
    MatCharac.write("\n\nThe matrix has a size " + str(Mat.size))
    MatCharac.write("\n\nchi^0(0, 0) = " +str(Mat[0, 0]))
    MatCharac.write("\n\nThe max abs real value of chi^0 is " + str(np.amax(np.abs(np.real(Mat)))))
    MatCharac.write("\n\nThe max abs imag value of chi^0 is " + str(np.amax(np.abs(np.imag(Mat)))))
    sum1 = np.sum(Mat)

    sum2 = np.sum(np.power(np.real(Mat), 2) + np.power(np.imag(Mat), 2))

    MatCharac.write("\n\nint_v1 int_v2 chi^0 dv1 dv2 = " + str(sum1))
    MatCharac.write("\n\nint_v1 int_v2 |chi^0|^2 dv1 dv2 = " + str(sum2))

    MatCharac.close()