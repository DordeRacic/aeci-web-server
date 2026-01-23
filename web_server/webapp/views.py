from django.shortcuts import render
from django import forms

# Create your views here.

class UploadFileForm(forms.Form):
    file = forms.FileField()

def get_model_response(file_bytes, filename):
    return f"{filename} successfully processed"

def upload_view(request):
    if "history" not in request.session:
        request.session["history"] = []

    output = None

    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file= request.FILES["file"]
            text = get_model_response(file.read(),file.name)
            output = text

            history = request.session["history"]
            history.append({"filename": file.name, "output" : text})
            request.session["history"] = history
        else:
            form= UploadFileForm()

        return render(request, "webapp/upload.html", {
            "form": form,
            "ocr_output": output,
            "history": request.session.get("history", [])
        })