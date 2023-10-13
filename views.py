from django.shortcuts import render, redirect
from .recommendation import get_recommendations
import pandas as pd
def home(request):
    data = pd.read_csv(r"C:\Users\ADMIN\Desktop\project\recomendation\csv file1.csv", error_bad_lines=False, encoding="latin-1")
    places = data['Place'].unique().tolist()
    return render(request, 'home.html', {'places': places})


def result(request):
    selected_place_name = request.POST['Place']
    recommendations = get_recommendations(selected_place_name)
    top_recommendations = recommendations[:11]  
    
    print("Selected Place:", selected_place_name)

    
    for i, recommendation in enumerate(top_recommendations):
        print("Recommendation {0}: {1}".format(i+1, recommendation))

    return render(request, 'result.html', {'selected_place_name': selected_place_name, 'recommendations': top_recommendations})
