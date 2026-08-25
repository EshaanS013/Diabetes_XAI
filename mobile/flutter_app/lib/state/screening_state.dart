import 'package:flutter/foundation.dart';

class ScreeningState extends ChangeNotifier {
  String audience = 'patient'; // or doctor
  bool loading = false;
  String? error;
  Map<String, dynamic>? lastResult;

  // Defaults are illustrative mid-range survey answers — not clinical advice.
  final Map<String, num> features = {
    'HighBP': 0,
    'HighChol': 0,
    'CholCheck': 1,
    'BMI': 27,
    'Smoker': 0,
    'Stroke': 0,
    'HeartDiseaseorAttack': 0,
    'PhysActivity': 1,
    'Fruits': 1,
    'Veggies': 1,
    'HvyAlcoholConsump': 0,
    'AnyHealthcare': 1,
    'NoDocbcCost': 0,
    'GenHlth': 3,
    'MentHlth': 0,
    'PhysHlth': 0,
    'DiffWalk': 0,
    'Sex': 1,
    'Age': 7,
    'Education': 4,
    'Income': 5,
  };

  void setAudience(String value) {
    audience = value;
    notifyListeners();
  }

  void setFeature(String key, num value) {
    features[key] = value;
    notifyListeners();
  }

  void setLoading(bool value) {
    loading = value;
    notifyListeners();
  }

  void setError(String? value) {
    error = value;
    notifyListeners();
  }

  void setResult(Map<String, dynamic>? value) {
    lastResult = value;
    notifyListeners();
  }
}
