// File Upload
// 
function ekUpload() {
	function Init() {
		console.log("Upload Initialised");

		var fileSelect = document.getElementById('file-upload'),
			fileDrag = document.getElementById('file-drag'),
			submitButton = document.getElementById('submit-button');

		fileSelect.addEventListener('change', fileSelectHandler, false);

		// Is XHR2 available?
		var xhr = new XMLHttpRequest();
		if (xhr.upload) {
			// File Drop
			fileDrag.addEventListener('dragover', fileDragHover, false);
			fileDrag.addEventListener('dragleave', fileDragHover, false);
			fileDrag.addEventListener('drop', fileSelectHandler, false);
		}
	}

	function fileDragHover(e) {
		var fileDrag = document.getElementById('file-drag');

		e.stopPropagation();
		e.preventDefault();

		fileDrag.className = (e.type === 'dragover' ? 'hover' : 'modal-body file-upload');
	}

	function fileSelectHandler(e) {
		// Fetch FileList object
		var files = e.target.files || e.dataTransfer.files;

		// Cancel event and hover styling
		fileDragHover(e);

		parseFile(files[0]);
		uploadFile(files[0]);
	}

	// Output
	function output(msg) {
		// Response
		var m = document.getElementById('messages');
		m.innerHTML = msg;
	}

	function parseFile(file) {

		console.log(file.name);
		output(
			'<strong>' + encodeURI(file.name) + '</strong>'
		);

		// var fileType = file.type;
		// console.log(fileType);
		var imageName = file.name;

		var isGood = (/\.(?=jpg|png|jpeg)/gi).test(imageName);
		if (isGood) {
			document.getElementById('start').classList.add("hidden");
			document.getElementById('response').classList.remove("hidden");
			document.getElementById('notimage').classList.add("hidden");
			// Thumbnail Preview
			document.getElementById('file-image').classList.remove("hidden");
			document.getElementById('file-image').src = URL.createObjectURL(file);
		} else {
			document.getElementById('file-image').classList.add("hidden");
			document.getElementById('notimage').classList.remove("hidden");
			document.getElementById('start').classList.remove("hidden");
			document.getElementById('response').classList.add("hidden");
			document.getElementById("file-upload-form").reset();
		}
	}

	function setProgressMaxValue(e) {
		var pBar = document.getElementById('file-progress');

		if (e.lengthComputable) {
			pBar.max = e.total;
		}
	}

	function updateFileProgress(e) {
		var pBar = document.getElementById('file-progress');

		if (e.lengthComputable) {
			pBar.value = e.loaded;
		}
	}

	function uploadFile(file) {
		var xhr = new XMLHttpRequest(),
		fileInput = document.getElementById('class-roster-file'),
		pBar = document.getElementById('file-progress'),
		fileSizeLimit = 1024; // In MB
		if (xhr.upload) {
			// Check if file is less than x MB
			if (file.size <= fileSizeLimit * 1024 * 1024) {
				// Progress bar
				pBar.style.display = 'inline';
				xhr.upload.addEventListener('loadstart', setProgressMaxValue, false);
				xhr.upload.addEventListener('progress', updateFileProgress, false);

				// File received / failed
				xhr.onreadystatechange = function(e) {
					if (xhr.readyState == 4) {
						// Everything is good!
						document.getElementById('file-progress').className = (xhr.status == 200 ? "success" : "failure");
						if (xhr.status == 200){
							window.location.href = xhr.response
						}
					}
				};

				const formData = new FormData();
				formData.append('image', file);

				// Start upload
				xhr.open('POST', "/upload", true);
				xhr.send(formData);
			} else {
				output('Please upload a smaller file (< ' + fileSizeLimit + ' MB).');
			}
		}
	}

	// Check for the various File API support.
	if (window.File && window.FileList && window.FileReader) {
		Init();
	} else {
		document.getElementById('file-drag').style.display = 'none';
	}
}
ekUpload();
