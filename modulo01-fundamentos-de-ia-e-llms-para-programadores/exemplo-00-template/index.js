import tf from '@tensorflow/tfjs-node';

async function treinarModelo(inputXs, outputYs) {
    // Criamos um modelo sequencial simples
    const model = tf.sequential();
    
    // Camada de entrada e oculta:
    // A camada de entrada tem 7 neurônios (correspondente às 7 características de entrada)
    // A camada oculta tem 80 neurônios e usa a função de ativação ReLU: 
    // ReLU (Rectified Linear Unit) é uma função de ativação que retorna o valor de entrada se for positivo, ou zero caso contrário.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' })); // Camada oculta
    
    // A camada de saída tem 3 neurônios (correspondente às 3 categorias de saída) e usa softmax para classificação (para proabilidade)
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // Camada de saída

    // Compilamos o modelo com otimizador Adam e função de perda categoricalCrossentropy (para classificação)
    // O que é Otimimizador Adam? É um algoritmo de otimização que ajusta os pesos do modelo durante o treinamento para minimizar a função de perda. 
    // Ele é eficiente e amplamente utilizado em redes neurais.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinamos o modelo usando os dados de entrada (inputXs) e saída (outputYs)
    // O número de épocas (epochs) define quantas vezes o modelo verá todo o conjunto de dados durante o treinamento.
    await model.fit(inputXs, outputYs, 
        {
            verbose: 0, // Para não mostrar o progresso do treinamento
            epochs: 100, // quantidade de vezes que o modelo verá todo o conjunto de dados
            shuffle: true, // Embaralha os dados a cada época para melhorar o treinamento
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
                }
            }
        }
    );

    return model;
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//inputXs.print();
//outputYs.print();

// Treinamos o modelo usando os dados de entrada (inputXs) e saída (outputYs)
// Quanto mais dados, melhor para o treinamento, mas para este exemplo simples, estamos usando apenas 3 amostras.
const modelo = await treinarModelo(inputXs, outputYs);
