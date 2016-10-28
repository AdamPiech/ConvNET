package adampiech.convnet.controll.engine.convNETLayers;

import org.opencv.core.Mat;

import java.util.Random;

/**
 * Created by Adam Piech on 2016-10-17.
 */
public class FullConnectedLayer {

    private Random rand = new Random();

    private Mat[] matData;
    private int[] intData;
    private int size;

    public FullConnectedLayer(Mat[] data, int size) {
        this.matData = data;
        this.size = size;
    }

    public FullConnectedLayer(int[] data, int size) {
        this.intData = data;
        this.size = size;
    }

    public int[] processLayerMatToInt() {
        int[] result = new int[size];
        for (int index = 0; index < size; index++) {
            result[index] = 0;
            for (int dataIndex = 0; dataIndex < matData.length; dataIndex++) {
                for (int indexX = 0; indexX < matData[0].width(); indexX++) {
                    for (int indexY = 0; indexY < matData[0].height(); indexY++) {
                        result[index] += matData[dataIndex].get(indexX, indexY)[0] * rand.nextDouble();
                    }
                }
            }
            result[index] += rand.nextInt();
            System.out.println("FullConnectedLayer_" + index + " " + result[index]);
        }
        return result;
    }

    public int[] processLayerIntToInt() {
        int[] result = new int[size];
        for (int index = 0; index < size; index++) {
            result[index] = 0;
            for (int dataIndex = 0; dataIndex < intData.length; dataIndex++) {
                result[index] += intData[dataIndex] * rand.nextDouble();
            }
            result[index] += rand.nextInt();
            System.out.println("FullConnectedLayer_" + index + " " + result[index]);
        }
        return result;
    }
}
