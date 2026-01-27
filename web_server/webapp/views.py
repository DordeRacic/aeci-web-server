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

            #Save file to disk
            #write.("Save", as: file_name)
            #file_loc = enter the location of the file
            #file_path = path to file {file_name}
            text = get_model_response(file.read(),file.name)
            output = text

            #Store file on DB (temporarily)

            #Create an OCR job entry

            #Dispatch job
            #dispatch_to_hpc(job.id, file_path)

            # Show job confirmation

            history = request.session["history"]
            history.append({"filename": file.name, "output" : text})
            request.session["history"] = history
        else:
            form= UploadFileForm()
    else:
        form= UploadFileForm()

    return render(request, "upload.html", {
        "form": form,
        "ocr_output": output,
        "history": request.session.get("history", [])
    })