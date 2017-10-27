import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class NeuralNetwork
{
    private String networkName;
    private MultiLayerNetwork thisNetwork;
    public NeuralNetwork(String name){
        this.networkName = name;
    }

    public void BuildNetwork(){
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
        thisNetwork = new MultiLayerNetwork(configuration);

    }

    public void TrainNetwork(){

    }

    public void TestNetwork(File testFile){
        File savedNetwork = new File(networkName);

        try {
            MultiLayerNetwork neuralNetwork = ModelSerializer.restoreMultiLayerNetwork(savedNetwork);
            NativeImageLoader loader = new NativeImageLoader(150,150,3);
            INDArray image = loader.asMatrix(testFile);
            DataNormalization scaler = new ImagePreProcessingScaler(0,1);
            scaler.transform(image);
            INDArray output = neuralNetwork.output(image);

            log.info("File is " + testFile);
            log.info("The neural nets prediction ##");
            log.info("list of probabilities per label ##");
            log.info("list of labels in order##");
            log.info(output.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void TestNetwork(List<File> files){
        for (File file: files
             ) {

        }
    }
}
