var model, canvas;
var classes = [];
var coordinates = [];
var mousedown = false;

$(function() {
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 0;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 10;
    canvas.renderAll();
    //setup listeners 
    canvas.on('mouse:up', function(e) {
        getFrame();
        mousedown = false
    });
    canvas.on('mouse:down', function(e) {
        mousedown = true
    });
    canvas.on('mouse:move', function(e) {
        recordCoor(e)
    });
})


/*
set the table of the predictions 
*/
function setTable(top5, probs) {
    //loop over the predictions 
    for (var i = 0; i < top5.length; i++) {
        let sym = document.getElementById('sym' + (i + 1))
        let prob = document.getElementById('prob' + (i + 1))
        sym.innerHTML = top5[i]
        prob.innerHTML = Math.round(probs[i] * 100)
    }

}


function recordCoor(event) {
    var pointer = canvas.getPointer(event.e);
    var posX = pointer.x;
    var posY = pointer.y;

    if (posX >= 0 && posY >= 0 && mousedown) {
        coordinates.push(pointer)
    }
}


function getMinBox() {
    var coorX = coordinates.map(function(p) {
        return p.x
    });
    var coorY = coordinates.map(function(p) {
        return p.y
    });

    var min_coordinates = {
        x: Math.min.apply(null, coorX),
        y: Math.min.apply(null, coorY)
    }
    var max_coordinates = {
        x: Math.max.apply(null, coorX),
        y: Math.max.apply(null, coorY)
    }

    //return as strucut 
    return {
        min: min_coordinates,
        max: max_coordinates
    }
}

function getImageData() {
    const mbb = getMinBox()

    const dpi = window.devicePixelRatio
    const imgData = canvas.contextContainer.getImageData(mbb.min.x * dpi, mbb.min.y * dpi,
                                                  (mbb.max.x - mbb.min.x) * dpi, (mbb.max.y - mbb.min.y) * dpi);
    return imgData
}


function getFrame() {
    if (coordinates.length >= 2) {

        const imgData = getImageData()

        const pred = model.predict(preprocess(imgData)).dataSync()
        const indices = findIndicesOfMax(pred, 5)
        const probs = findTopValues(pred, 5)
        const names = getclasses(indices)
        setTable(names, probs)
    }

}

function getclasses(indices) {
    var outp = []
    for (var i = 0; i < indices.length; i++)
        outp[i] = classes[indices[i]]
    return outp
}

async function loadDict() {
    loc = 'model_tfjs/class_names.txt'
    
    await $.ajax({
        url: loc,
        dataType: 'text',
    }).done(success);
}

function success(data) {
    const lst = data.split(/\n/)
    for (var i = 0; i < lst.length - 1; i++) {
        let symbol = lst[i]
        classes[i] = symbol
    }
}

function findIndicesOfMax(inp, count) {
    var outp = [];
    for (var i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function(a, b) {
                return inp[b] - inp[a];
            }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}

function findTopValues(inp, count) {
    var outp = [];
    let indices = findIndicesOfMax(inp, count)
    // show 5 greatest scores
    for (var i = 0; i < indices.length; i++)
        outp[i] = inp[indices[i]]
    return outp
}



function preprocess(imgData) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgData, numChannels = 1)
        
        const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat()
        
        const offset = tf.scalar(255.0);
        const normalized = tf.scalar(1.0).sub(resized.div(offset));

        const batched = normalized.expandDims(0)
        return batched
    })
}



async function start() {
    model = await tf.loadLayersModel('model_tfjs/model.json') 
    model.predict(tf.zeros([1, 28, 28, 1]))
    allowDrawing()
    await loadDict()
}


function allowDrawing() {
    canvas.isDrawingMode = 1;
    document.getElementById('status').innerHTML = 'Model Loaded';
    $('button').prop('disabled', false);
    var slider = document.getElementById('myRange');
    slider.oninput = function() {
        canvas.freeDrawingBrush.width = this.value;
    };
}


function erase() {
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    coordinates = [];
}