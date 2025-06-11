$(document).ready(function () {
  const modelMap = {
    "chatgpt-4o-latest": "GPT 4o",
    "gpt-4-turbo-preview": "GPT 4",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "claude-3-sonnet-20240229": "Claude 3 Sonnet",
  };

  if ("SpeechRecognition" in window || "webkitSpeechRecognition" in window) {
    $("#micButton").addClass("d-flex").show();
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.lang = "en-US";
    recognition.interimResults = true;
    const prompt = document.getElementById("prompt");

    $("#micButton").click(function () {
      prompt.textContent = "Listening...";
      recognition.start();
    });

    recognition.onresult = (event) => {
      prompt.textContent = event.results[0][0].transcript;
    };

    recognition.onaudioend = (event) => {
      if (prompt.textContent === "Listening...") {
        prompt.textContent = "";
      }
      $("#generateButton").trigger("click");
    };

    recognition.onerror = (event) => {
      console.error("Error occurred in recognition: " + event.error);
    };
  } else {
    $("#micButton").hide();
  }

  $("#darkModeSwitch").change(function () {
    if ($(this).is(":checked")) {
      $("html").attr("data-bs-theme", "dark");
    } else {
      $("html").attr("data-bs-theme", "light");
    }
  });

  $("#loadButton").click((event) => {
    event.preventDefault();

    if (!$("#url").val()) {
      showMessage("error", "Youtube URL is required!");
      return;
    }

    addSpinner("#loadButton");

    $.ajax({
      type: "POST",
      url: "/load",
      data: {
        url: $("#url").val(),
      },
      success: function (response) {
        showMessage("success", "Video loaded successfully!");
      },
      error: function (xhr, status, error) {
        showMessage("error", xhr.responseText);
      },
      complete: function () {
        removeSpinner("#loadButton");
      },
    });
  });

  $("#generateButton").click((event) => {
    event.preventDefault();

    const prompt = $("#prompt").val();
    const selectedModels = $("input[name='models']:checked")
      .map(function () {
        return this.value;
      })
      .get();

    if (!prompt) {
      showMessage("error", "Prompt is required!");
      return;
    }

    if (selectedModels.length === 0) {
      showMessage("error", "Please select at least one model!");
      return;
    }

    addSpinner("#generateButton");

    $("#responseDiv").empty();

    streamDataForAllModels(prompt, selectedModels);
  });

  function streamDataForAllModels(prompt, selectedModels) {
    const promises = selectedModels.map((model) =>
      streamData(prompt, model).catch((error) => {
        console.log(`Error for model ${model}: ${error.message}`);
        showMessage("error", `Error for model ${model}: ${error.message}`);
        return null; // Prevent it from halting other requests
      })
    );

    Promise.allSettled(promises)
      .then(() => {
        removeSpinner("#generateButton");
      })
      .catch((error) => {
        console.log(error);
        showMessage("error", error);
        removeSpinner("#generateButton");
      });
  }

  function streamData(prompt, model) {
    const url = "/generate";
    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("model", model);
    modelResponseContainer = `<div id="div_${model}" name="div_${model}" style="display:none;">
            <label class="form-label ms-1" id="label_${model}" name="label_${model}"><b>${modelMap[model]}</b></label>
            <pre class="form-control border-2 border-secondary" id="response_${model}"
              name="response_${model}"
              style="white-space: pre-wrap; font-family: sans-serif;"></pre>
          </div>`;
    $("#responseDiv").append(modelResponseContainer);

    return fetch(url, {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          return response.text().then((errorMessage) => {
            throw new Error(errorMessage);
          });
        }
        const reader = response.body.getReader();
        const textDecoder = new TextDecoder();

        const processChunk = ({ value, done }) => {
          if (done) {
            $(`#response_${model}`).html(
              $(`#response_${model}`)
                .html()
                .replaceAll(" **", " <b>")
                .replaceAll("**", "</b>")
            );
            return;
          }

          const text = textDecoder.decode(value);
          $(`#response_${model}`).append(text);
          reader.read().then(processChunk);
        };

        $(`#div_${model}`).show();
        reader.read().then(processChunk);
      })
      .catch((error) => {
        console.log(error);
        showMessage("error", `Error for model ${model}: ${error.message}`);
      });
  }

  function showMessage(status, message) {
    if (status === "error") {
      $("h4#modalHeading")
        .text("Error")
        .removeClass("text-success")
        .addClass("text-danger");
    } else if (status === "success") {
      $("h4#modalHeading")
        .text("Success")
        .removeClass("text-danger")
        .addClass("text-success");
    }
    $("#modalContent").text(message);
    $("#openModal").trigger("click");
  }

  function addSpinner(element) {
    $(element).prop("disabled", true);
    $(element).find("#buttonText").hide();
    $(element).find("#buttonSpinner").show();
  }

  function removeSpinner(element) {
    $(element).find("#buttonText").show();
    $(element).find("#buttonSpinner").hide();
    $(element).prop("disabled", false);
  }
});
