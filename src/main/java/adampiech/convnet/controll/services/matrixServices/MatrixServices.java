package adampiech.convnet.controll.services.matrixServices;

import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.opencv.core.Mat;

import java.util.List;

import static org.opencv.core.CvType.CV_16S;

/**
 * Created by Adam Piech on 2016-11-14.
 */

public class MatrixServices {

    public static Matrix[] matToMatrix(Mat[] mat) {

        int length = mat.length;
        Matrix[] matrix = new Matrix[length];

        for (int depth = 0; depth < length; depth++) {
            matrix[depth] = matToMatrix(mat[depth]);
        }

        return matrix;
    }

    public static Matrix[] matToMatrix(List<Mat> mat) {

        int length = mat.size();
        Matrix[] matrix = new Matrix[length];

        for (int depth = 0; depth < length; depth++) {
            matrix[depth] = matToMatrix(mat.get(depth));
        }

        return matrix;
    }

    public static Matrix matToMatrix(Mat mat) {

        int width = mat.width();
        int height = mat.height();

        Matrix matrix = new Basic2DMatrix(width, height);
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                double value = mat.get(row, col)[0] / 255.0;
                matrix.set(row, col, value);
            }
        }

        return matrix;
    }

    public static Mat[] matrixToMat(Matrix[] matrix) {

        int length = matrix.length;
        Mat[] mat = new Mat[length];

        for (int depth = 0; depth < length; depth++) {
            mat[depth] = matrixToMat(matrix[depth]);
        }

        return mat;
    }

    public static Mat matrixToMat(Matrix matrix) {

        int width = matrix.rows();
        int height = matrix.columns();

        Mat mat = new Mat(width, height, CV_16S);
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                double value = matrix.get(row, col) * 255.0;
                mat.put(row, col, value);
            }
        }

        return mat;
    }

    public static double arrayMultiplicationAndSum(Matrix m1, Matrix m2) {
        double result = 0.0;
        if (m1.rows() == m2.rows() && m1.columns() == m2.columns()) {
            for (int row = 0; row < m1.rows(); row++) {
                for (int col = 0; col < m1.columns(); col++) {
                    result += m1.get(row, col) * m2.get(row, col);
                }
            }
        }
        return result;
    }

    public static Matrix arrayMultiplication(Matrix m1, Matrix m2) {
        Matrix result = null;
        if (m1.rows() == m2.rows() && m1.columns() == m2.columns()) {
            result = new Basic2DMatrix(m1.rows(), m1.columns());
            for (int row = 0; row < m1.rows(); row++) {
                for (int col = 0; col < m1.columns(); col++) {
                    result.set(row, col, m1.get(row, col) * m2.get(row, col));
                }
            }
        }
        return result;
    }

    public static Matrix arraySum(Matrix m1, Matrix m2) {
        Matrix result = null;
        if (m1.rows() == m2.rows() && m1.columns() == m2.columns()) {
            result = new Basic2DMatrix(m1.rows(), m1.columns());
            for (int row = 0; row < m1.rows(); row++) {
                for (int col = 0; col < m1.columns(); col++) {
                    result.set(row, col, m1.get(row, col) + m2.get(row, col));
                }
            }
        }
        return result;
    }

    public static double arrayMultiplicationAndSum(Mat m1, Mat m2) {
        double result = 0.0;
        if (m1.width() == m2.width() && m1.height() == m2.height()) {
            for (int row = 0; row < m1.width(); row++) {
                for (int col = 0; col < m1.height(); col++) {
                    result += m1.get(row, col)[0] * m2.get(row, col)[0];
                }
            }
        }
        return result;
    }

    public static Matrix[] subMatrix(Matrix[] m, int rowStart, int colStart, int rowEnd, int colEnd) {
        Matrix[] newMatrix = new Matrix[m.length];
        for (int index = 0; index < m.length; index++) {
            newMatrix[index] = m[index].slice(rowStart, colStart, rowEnd, colEnd);
        }
        return newMatrix;
    }

    public static Matrix[] copyMatrix(Matrix[] m) {
        Matrix[] newMatrix = new Matrix[m.length];
        for (int depth = 0; depth < m.length; depth++) {
            newMatrix[depth] = new Basic2DMatrix(m[depth].rows(), m[depth].columns());
            for (int row = 0; row < m[depth].rows(); row++) {
                for (int col = 0; col < m[depth].columns(); col++) {
                    newMatrix[depth].set(row, col, m[depth].get(row, col));
                }
            }
        }
        return newMatrix;
    }

    public static Matrix[] copyMatrixArchitecture(Matrix[] m) {
        Matrix[] newMatrix = new Matrix[m.length];
        for (int index = 0; index < m.length; index++) {
            newMatrix[index] = Matrix.zero(m[index].rows(), m[index].columns());
        }
        return newMatrix;
    }

    public static boolean matricesOwnSameSize(Matrix[] m1, Matrix[] m2) {
        if (m1.length != m2.length) {
            return false;
        }
        return matricesOwnSameSize(m1[0], m2[0]);
    }

    public static boolean matricesOwnSameSize(Matrix m1, Matrix m2) {
        if (m1.rows() != m2.rows()) {
            return false;
        }
        if (m1.columns() != m2.columns()) {
            return false;
        }
        return true;
    }

}
