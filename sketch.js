


const hands = [//'🤲',
//'👐',
//'🙌',
//'👏',
//'🤝',
//'👍',
//'👎',
//'👊',
'✊',
//'🤛',
//'🤜',
//'🤞',
//'✌️',
//'🤟',
//'🤘',
//'👌',
//'👈',
//'👉',
//'👆',
//'👇',
//'☝️',
//'✋',
'🤚'
//'🖐',
//'🖖',
//'👋',
//'🤙',
//'💪',
//'🖕'
];


var video;
const SNAPSHOT_PX = "32px";
var MAX_TRAIN_CAPTURE = 64;

var training = false;

var samples = [];

const CAP_SIZE = {w:224,h:224};

var errorMessage;
var label;
var traininIterationLabel;
let containerImages ;
var divArray = [];
var fileInput;
var predictButton;
var captureButton;
var emojiPredictedDiv;
var predictButton;

function setup() {

	let canvas = createCanvas(400,400);
	canvas.id("myChart");
	canvas.width = "500px"
	canvas.height = "500px"

	var constraints = {
		audio: false,
		video: {
			mandatory: {
				minWidth: CAP_SIZE.w,
				minHeight: CAP_SIZE.h
			},
			optional: [{ maxFrameRate: 10 }]
		}
	};

	//workaround to make p5js createCapture work on firefox.
	navigator.getUserMedia = navigator.getUserMedia       ||
                               navigator.webkitGetUserMedia ||
                               navigator.mozGetUserMedia    ||
                               undefined;


                             
	video = createCapture(constraints);
	video.elt.width = CAP_SIZE.w;
	video.elt.height = CAP_SIZE.h;
	video.style("margin","auto");
	video.parent("left-container");

	captureButton = createButton("Capture").mousePressed(prepareTrain);
	let trainButton = createButton("Start Training").mousePressed(startTrain);
	predictButton = createButton("Predecit").mousePressed(predict);

	captureButton.parent("left-container");
	trainButton.parent("left-container");
	predictButton.parent("left-container");

	//container shown on top on camera to show the predicter emoji
	emojiPredictedDiv = createDiv();
	emojiPredictedDiv.position(video.position().x, video.position().y);
	emojiPredictedDiv.style("z-index","10");
	emojiPredictedDiv.style("font-size","150px");
	emojiPredictedDiv.style("opacity","0.5");
	emojiPredictedDiv.style("margin","auto");
	emojiPredictedDiv.parent("left-container");
	emojiPredictedDiv.elt.innerHTML = ""

	//container for snapshots
	containerImages = createDiv();

	//error messages TBD
	errorMessage = createP("Please select a hand you want to train");
	errorMessage.hide();


	//create buttons with the handemoji 

	selector = createDiv();
	selector.parent("target-container")
	selector.id("selector");
	selector.hide();

	let instr = createButton("instructions");
	instr.style("margin","20px");
	instr.parent("target-container");
	instr.mousePressed(handSelected);
	instr.idemoji = "instructions";

	for(let i=0;i<hands.length;i++){
		let hand = hands[i];
		let button = createButton(hand);
		button.style("margin","20px");
		button.mousePressed(handSelected);
		button.parent("target-container");
		button.idemoji = hand;

		divImageContainer = createDiv();
		//divImageContainer.style("margin","20px");
		divImageContainer.addClass("snapshot-container");
		divImageContainer.hide();
		divImageContainer.parent("right-container");
		divArray[hand] = divImageContainer;
	}


	let d = document.getElementById("instructions");
	divImageContainer = createDiv();
		divImageContainer.child(d);
		divImageContainer.parent("right-container");
	divArray["instructions"] = divImageContainer;
	//selector.position(Object.values(divArray)[0].position().x,Object.values(divArray)[0].position().y);
	//presetup the model
	buildModel();
}

var m; //my machine
async function buildModel(){
	m = new Machine();
	m.build(hands.length);
}


var prepareTrain = async function(){
	
	if(training) return;
	training = true;

	if(label === undefined){
		errorMessage.show();
		training = false;
		return;
	}

	errorMessage.hide();

	let labelCaptured = label;
	for(let i=0;i<MAX_TRAIN_CAPTURE;i++){
		setTimeout(function(){
			let data = getCapture(labelCaptured);
				//add all the training images the user want
				samples.push(data);
				if(i == MAX_TRAIN_CAPTURE -1){
					training = false;
				} 
			},50);
	}

}


var getCapture = function(label){
	let snapshotcanvas = video.get().canvas;
	let imagedata = snapshotcanvas.getContext('2d').getImageData(0,0,CAP_SIZE.w,CAP_SIZE.h);
	
	createSnapshot(snapshotcanvas,label)

	let data = removeAlphaNormalize(imagedata.data);

	//assign the label
	data.label = hands.indexOf(label);

	return data;
}


var createSnapshot = function(snapshotcanvas,label){
	//show image
	let dataurl = snapshotcanvas.toDataURL("image/png");
	let img = createImg(dataurl);
	img.attribute("width", SNAPSHOT_PX);
	img.attribute("height", SNAPSHOT_PX );
	img.id("ImgSample");
	img.parent(divArray[label]);
	img.elt.onmouseover = function(){ this.classList.add("big-img-sample"); }
	img.elt.onmouseout = function(){ this.classList.remove("big-img-sample"); }
}

var handSelected = function(){
	errorMessage.hide();

	label = this.idemoji;
	for (key in divArray) {
		let div = divArray[key];
		if(div === divArray[label]) div.show();
		else div.hide();
	}

	//move the triagnle selector to selected button
	selector.position(this.position().x,this.position().y + 38);
	selector.show();
}


function draw(){
	//nothing for the moment.
	if(isPredicting) predictInternal();
}


/*The tensorflow part*/

const PERCENTAGE_TEST = 20; //20% of all the input will be for testing
const TRAIN_BATCHES = 10;
const BATCH_SIZE = MAX_TRAIN_CAPTURE*2;

var trainIndices ;

function removeAlphaNormalize(imagedata){
	let imageBuffer = new Float32Array(CAP_SIZE.w*CAP_SIZE.h*3);
	for(let j = 0; j<CAP_SIZE.w*CAP_SIZE.h;j++) {
		let outIndex = j*3;
		let sourceIndex = j*4;
		imageBuffer[outIndex]=(imagedata[sourceIndex]/127 )-1; //R
		imageBuffer[outIndex+1]=((imagedata[sourceIndex+1]/127)-1); //G
		imageBuffer[outIndex+2]=((imagedata[sourceIndex+2]/127)-1); //B
	}

	return imageBuffer;
}


var isTraining = false;
var startTrain = async function() {

	let e = ui.getEpoche();
	m.setEpoche(e);

	let l = ui.getLearningRate();
	m.setLearningRate(l);

	if(isTraining) return;
	isTraining = true;

	var dataChart = [];
	try{

		//let trainIndices = tf.util.createShuffledIndices(samples.length);
		
		var labels = []
		myXs = [];
		for (var i =  0; i<samples.length; i++) {
			console.log(tf.memory().numBytes);
			//let index = trainIndices[i]; //shuffled in model.fit
			imagedata = samples[i];
			myXs.push(m.getTensorXs(imagedata));
		    //myXs.push( tf.tidy(() => { let t = createTensor(imagedata); return mobilenet.predict(t); }  ) ) ;
		    labels.push(imagedata.label);
		}
		
		let tfLabels = tf.tensor1d(labels, 'int32');
		//normalize the data so whatever the int of an emoji we normalize it between 0 and 1
		ys = tf.oneHot(tfLabels, hands.length).cast('int32');
		var xs = tf.concat(myXs);

		tfLabels.dispose();
		myXs.forEach(function(v){ v.dispose();});

		if (xs == null) {
			throw new Error('Add some examples before training!');
		}

	   //if (!(samples.length > 0)) {
	   // throw new Error(
	   //     `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
	   // }

	   // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
	   m.train(xs, ys, {
			      onEpochEnd: async (batch, logs) => {
			        console.log('Loss: ' + logs.loss.toFixed(5) +" batch " +batch);
			      	dataChart.push({x:batch,y:logs.loss.toFixed(5)})
			        await tf.nextFrame();
			      },
			      onTrainEnd: () => { xs.dispose();
			  						ys.dispose();
									this.isTraining = false;
									ui.showChartLoss(dataChart);
			  					}
			    });

	}catch(e){
		console.error(e);
		isTraining = false;
	}

}

var isPredicting = false;
var predict = async function () {
	if(isPredicting) {
		isPredicting = false;
		predictButton.elt.innerHTML = "Predict";
	} else {

		isPredicting = true;
	 	predictButton.elt.innerHTML = "Predicting....";
 		ui.setDetectedEmoji("");
 	 }

}

var predictInternal = function (){
	let snapshotcanvas = video.get().canvas;
	let imagedata = snapshotcanvas.getContext('2d').getImageData(0,0,CAP_SIZE.w,CAP_SIZE.h);
	let img = removeAlphaNormalize(imagedata.data);

	const predictedClass = m.predict(m.getTensorXs(img));

	const classId =  predictedClass[0];
    //predictedClass.dispose();

    ui.setDetectedEmoji(hands[classId]);
}

//check the removeAlphaNormalize is working. We paint the processed image in a canvas
var test = function(){
	var img = removeAlphaNormalize(samples[0].data);
	loadPixels();
	for(let i = 0;i<CAP_SIZE.w*CAP_SIZE.h;i++){
		let c = color(img[i*3]*127, img[i*3+1]*127, img[i*3+2]*127);
		//console.log(aaa[i*3]*255 +""+ aaa[i*3+1]*255+""+ aaa[i*3+2]*255);
		set(i%CAP_SIZE.w,i/CAP_SIZE.w, c) ;
	}
	updatePixels();
}




//Ui temporary object
var ui = {};

ui.getEpoche = function(){
	return document.getElementById("epoche").value;
}

ui.getLearningRate = function(){
	return document.getElementById("learning-rate").value;	
}

ui.setDetectedEmoji = function(emoji){
	emojiPredictedDiv.elt.innerHTML = emoji;
}

ui.showChartLoss = function(data){
	var ctx = document.getElementById("myChart");
	var myChart = new Chart(ctx, {
    type: 'line',
    data: {
        datasets: [{
            label: 'Loss per Epoche',
            data: data,
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero:true
                }
            }]
        }
    }
});
}

