'use strict'

const EPOCHE = 20;
//const LEARNING_RATE = 0.0001;

const LEARNING_RATE = 0.0001;
const OPTIMIZER = "ADAM";

class Machine{
	
	constructor(){
		this.epoche = EPOCHE;
		this.learningRate = LEARNING_RATE;
		this.isTraining = false;
		this.setOptimizer(OPTIMIZER);
	}

	setEpoche(epoche){
		let e = parseInt(epoche);
		if(e === NaN) throw new Error("Epoche must be a number");
		this.epoche = e;
	}

	setLearningRate(learningRate){
		let e = parseFloat(learningRate);
		if(e === NaN) throw new Error("Learning rate must be a number");
		this.learningRate = e;
	}

	setOptimizer(optimizer){

		if(optimizer === "ADAM"){
			this.optimizer = tf.train.adam(LEARNING_RATE);
		}else if(otimizer == "SGD"){
			this.optimizer = tf.train.sgd(LEARNING_RATE);
		}
	}


	async  loadMobilenet() {
	   const mobilenet = await tf.loadModel(
	      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

	  // Return a model that outputs an internal activation.
	  const layer = mobilenet.getLayer('conv_pw_13_relu');
	  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
	};

	async build(numCategories){
 		
 		if(numCategories === undefined || numCategories ===null) throw  new Error("Please specify number of categorization of your machine");
 		if(!Number.isInteger(numCategories))  throw  new Error("Parameter numCategories has to be a number");
 		
 		this.numCategories = numCategories;

 		//load the mobilenet
	 	this.mobilenet = await this.loadMobilenet();

	 	//create our own mchine on top of the mobilenet
	 	this.model = tf.sequential();
		this.model.add(tf.layers.flatten({inputShape: [7, 7, 256]})); //[7, 7, 256] is the output of mobilenet so our input

		//fully conected layer
		this.model.add(tf.layers.dense({
		  units: this.numCategories, //number of emojis to recognize
		  kernelInitializer: 'VarianceScaling',
		  activation: 'softmax'
		}));

		this.model.compile({
		  optimizer: this.optimizer,
		  loss: 'categoricalCrossentropy',
		  metrics: ['accuracy'],
		});
	}


	getTensorXs(imagedata){
		let v = this.createTensor(imagedata);
		let p = this.mobilenet.predict(v);
		v.dispose();
		return p;
	}


	createTensor(data){
	 return tf.tensor4d(data, [1,CAP_SIZE.w,CAP_SIZE.h,3],'float32');
	}

	async train(xs,ys,callbacks){

		if (xs == null) {
			    throw new Error('Please follow the instructions!');
		 }

		if(this.isTraining) { console.log("Machine is still training"); return;}
		this.isTraining = true;
		
		try{
	
			  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
			  this.model.fit(xs, ys, {
			    batchSize: xs.shape[0], //TODO make it parameter
			    epochs: this.epoche,
			    validationSlplit:0.2,
			    shuffle: true,
			    callbacks: callbacks
			  });
		}catch(e){
			console.error(e);
			this.isTraining = false;
		}
	}

	predict(activation){
 		return tf.tidy(() => {
		
	   		 const output = m.model.predict(activation);

		    const axis = 1;
	  	  	return  output.argMax(axis).dataSync();
	  		});
	}

}