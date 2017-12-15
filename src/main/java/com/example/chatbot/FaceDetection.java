package com.example.chatbot;

        import java.io.File;
        import java.io.IOException;
        import java.net.URL;

        import org.apache.commons.io.FileUtils;
        import org.bytedeco.javacpp.Loader;
        import static org.bytedeco.javacpp.opencv_core.*;
        import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
        import static org.bytedeco.javacpp.opencv_objdetect.*;

public final class FaceDetection {

    private final CvHaarClassifierCascade Classifier;
    private final CvMemStorage Storage;

    public FaceDetection() throws IOException {
        File file = null;
        try {
            URL url = new URL("https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml");
            file = new File("front_face.xml");
            FileUtils.copyURLToFile(url, file);
        } catch (Exception e) {
            e.printStackTrace();
        }                           //Download the front_face classifier file to detect faces

        Classifier = new CvHaarClassifierCascade(cvLoad(file.getAbsolutePath()));
        file.delete();

        Storage = CvMemStorage.create();
    }

    public int numFacesInImage(String imagePath) {
        final IplImage image = cvLoadImage(imagePath);          //Load image for face detection
        final CvSeq faces = cvHaarDetectObjects(image, Classifier, Storage,1.1,3, CV_HAAR_DO_CANNY_PRUNING);

        cvClearMemStorage(Storage);

        return faces.total();
    }
}