%%Carga de Datos
[XTrain, YTrain, XTest, YTest] = load_train1000('mnist');
%% Visualización de DATOS

ims = randperm(1000,20);

figure(1) 

for i=1:length(ims)
    subplot(4,5,i)
    imagen = XTrain(:,:,:,ims(i));
    imshow(imagen,[28,28])
    etiqueta = YTrain(ims(i));
    title(etiqueta) 
end

%% Arquitectura de la red

nclasses = 10; %salidas
[x,y,ch,N]= size(XTrain); 
%[x,y,ch,~]= size(XTrain);  Si N no interesa sacar
nfeatures = 2^8;
layers = [...
    
    imageInputLayer([x y ch], 'Name', 'Capa de entrada')
    fullyConnectedLayer(nfeatures,'Name', 'Capa oculta') %Conecta las capas
    %Función de activación ~ Relu
    reluLayer('Name', 'Relu')
    fullyConnectedLayer(nclasses,'Name', 'Capa de salida')%Conecta las capas
    %Función de activación
    softmaxLayer('Name','Softmax')
    classificationLayer('Name','Clases')

];
grafo_capas = layerGraph(layers);
figure(2)
plot(grafo_capas)
%% Entrenamiento [Easy]
options = trainingOptions('adam',...
                            'Shuffle','every-epoch', ...
                            'MaxEpochs', 100, ...
                            'MiniBatchSize', 10, ...
                            'ValidationData', {XTest, YTest},...
                            'ValidationFrequency',10,...
                            'Plots', 'training-progress');
net = trainNetwork(XTrain, YTrain, layers, options);
%% Validación

YTrain_pred = predict(net, XTrain);

[M,I] = max(YTrain_pred,[],2);
I = categorical(I-1);
%Comparar con el vector verdadero XTrain
    %Confusion Matrix
    CTrain = confusionmat(YTrain,I);
    figure(3)
    confusionchart(CTrain)
    
    %Part 2
    
    YTest_pred = predict(net, XTest);

[M,I] = max(YTest_pred,[],2);
YTest_pred = categorical(I-1);
%Comparar con el vector verdadero XTrain
    %Confusion Matrix
    CTest = confusionmat(YTest,YTest_pred);
    figure(4)
    confusionchart(CTest)

