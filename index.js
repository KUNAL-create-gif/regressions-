const tf = require('@tensorflow/tfjs');
const LinearRegression = require('./linear-regression');
const loadCSV = require('./load-csv');
const plot = require('node-remote-plot');


let { features, labels, testFeatures, testLables } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 100,
    batchsize: 10
});



regression.train();
const r2 = regression.test(testFeatures, testLables);

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLable: 'Mean Squared Error'
});


console.log('R2 is', r2);

regression.predict([
    [120, 2, 380],
    [135, 2.1, 420]
]).print();
