document.addEventListener("DOMContentLoaded", function() {
    var inputFile = document.getElementById("file");
    var inputVideo = document.getElementById("input-video");
    var processedVideo = document.getElementById("processed-video");
    var plotCanvas = document.getElementById("plot-canvas");

    // Add an event listener to the file input element
    inputFile.addEventListener("change", function() {
        // Create a FormData object to send the file to the API
        var formData = new FormData();
        formData.append("file", inputFile.files[0]);

        // Send a POST request to the API with the file data
        fetch("/", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Set the source of the input video
            inputVideo.src = data.input_video;

            // Set the source of the processed video
            processedVideo.src = data.processed_video;

            // Draw the plot on the canvas
            drawPlot(data.plot_data);
        })
        .catch(error => console.log(error));
    });

    function drawPlot(plotData) {
        var ctx = plotCanvas.getContext("2d");
        var width = plotCanvas.width;
        var height = plotCanvas.height;

        // Clear the canvas
        ctx.clearRect(0, 0, width, height);

        // Draw the plot data
        ctx.beginPath();
        ctx.moveTo(0, height - plotData[0]);

        for (var i = 1; i < plotData.length; i++) {
            var x = (i / plotData.length) * width;
            var y = height - plotData[i];
            ctx.lineTo(x, y);
        }

        ctx.strokeStyle = "#333333";
        ctx.lineWidth = 2;
        ctx.stroke();
    }
});