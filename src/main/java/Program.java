import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Program {
    private static Logger log = LoggerFactory.getLogger(Program.class);

    public static void main(String[] args) throws IOException {
        int height = 150;
        int width = 150;
        int channels = 3;
        int rngSeed = 123;
        Random randomNrGenerator = new Random(rngSeed);
        int batchSize = 128;
        int outputNr = 2;
        int numEpochs = 15;

        File trainPath = new File("C:/Users/palmi/Desktop/NeuralNetwork/TrainSet");
        File testPath = new File("C:/Users/palmi/Desktop/NeuralNetwork/TestSet");

        FileSplit train = new FileSplit(trainPath, NativeImageLoader.ALLOWED_FORMATS,randomNrGenerator);
        FileSplit test = new FileSplit(testPath,NativeImageLoader.ALLOWED_FORMATS,randomNrGenerator);



        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelGenerator);

        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIterator = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNr);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIterator);
        dataIterator.setPreProcessor(scaler);

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(height * width * channels)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(outputNr)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height,width,channels))
                .build();

        MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
        neuralNetwork.init();

        for(int i = 0; i< numEpochs; i++){
            System.out.println(i);
            neuralNetwork.fit(dataIterator);
        }

        File locationToSave = new File("trained_car-people_model.zip");

        boolean saveUpdater = false;

        ModelSerializer.writeModel(neuralNetwork,locationToSave,saveUpdater);
        /*
        recordReader.reset();
        recordReader.initialize(test);
        DataSetIterator testIte = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNr);
        scaler.fit(testIte);
        testIte.setPreProcessor(scaler);

        Evaluation eval = new Evaluation(outputNr);

        while(testIte.hasNext()){
            DataSet next = testIte.next();
            INDArray output = neuralNetwork.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(),output);
        }
        log.info(eval.stats());
        */
    }



}
