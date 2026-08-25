import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/api_client.dart';
import '../state/screening_state.dart';
import 'doctor_screen.dart';
import 'patient_screen.dart';

class QuestionnaireScreen extends StatelessWidget {
  const QuestionnaireScreen({super.key});

  static const binaryFields = <String, String>{
    'HighBP': 'High blood pressure history',
    'HighChol': 'High cholesterol history',
    'CholCheck': 'Cholesterol checked (5y)',
    'Smoker': 'Smoked ≥100 cigarettes',
    'Stroke': 'History of stroke',
    'HeartDiseaseorAttack': 'CHD / heart attack',
    'PhysActivity': 'Physical activity (30d)',
    'Fruits': 'Fruit ≥1x / day',
    'Veggies': 'Vegetables ≥1x / day',
    'HvyAlcoholConsump': 'Heavy alcohol use',
    'AnyHealthcare': 'Has healthcare coverage',
    'NoDocbcCost': 'Could not see doctor (cost)',
    'DiffWalk': 'Difficulty walking',
    'Sex': 'Sex (0=female, 1=male)',
  };

  @override
  Widget build(BuildContext context) {
    final state = context.watch<ScreeningState>();
    return Scaffold(
      appBar: AppBar(title: const Text('Health questionnaire')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            'Self-reportable indicators only. Not a diagnosis.',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: 12),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'patient', label: Text('Patient')),
              ButtonSegment(value: 'doctor', label: Text('Doctor')),
            ],
            selected: {state.audience},
            onSelectionChanged: (s) => state.setAudience(s.first),
          ),
          const SizedBox(height: 16),
          _SliderField(
            label: 'BMI',
            value: state.features['BMI']!.toDouble(),
            min: 12,
            max: 60,
            divisions: 48,
            onChanged: (v) => state.setFeature('BMI', v.roundToDouble()),
          ),
          _SliderField(
            label: 'General health (1=excellent … 5=poor)',
            value: state.features['GenHlth']!.toDouble(),
            min: 1,
            max: 5,
            divisions: 4,
            onChanged: (v) => state.setFeature('GenHlth', v.round()),
          ),
          _SliderField(
            label: 'Age category (1–13)',
            value: state.features['Age']!.toDouble(),
            min: 1,
            max: 13,
            divisions: 12,
            onChanged: (v) => state.setFeature('Age', v.round()),
          ),
          _SliderField(
            label: 'Education (1–6)',
            value: state.features['Education']!.toDouble(),
            min: 1,
            max: 6,
            divisions: 5,
            onChanged: (v) => state.setFeature('Education', v.round()),
          ),
          _SliderField(
            label: 'Income (1–8)',
            value: state.features['Income']!.toDouble(),
            min: 1,
            max: 8,
            divisions: 7,
            onChanged: (v) => state.setFeature('Income', v.round()),
          ),
          _SliderField(
            label: 'Poor mental-health days (0–30)',
            value: state.features['MentHlth']!.toDouble(),
            min: 0,
            max: 30,
            divisions: 30,
            onChanged: (v) => state.setFeature('MentHlth', v.round()),
          ),
          _SliderField(
            label: 'Poor physical-health days (0–30)',
            value: state.features['PhysHlth']!.toDouble(),
            min: 0,
            max: 30,
            divisions: 30,
            onChanged: (v) => state.setFeature('PhysHlth', v.round()),
          ),
          const Divider(),
          ...binaryFields.entries.map((e) {
            final on = state.features[e.key] == 1;
            return SwitchListTile(
              title: Text(e.value),
              value: on,
              onChanged: (v) => state.setFeature(e.key, v ? 1 : 0),
            );
          }),
          const SizedBox(height: 16),
          if (state.error != null)
            Text(state.error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          FilledButton(
            onPressed: state.loading
                ? null
                : () async {
                    final api = context.read<ApiClient>();
                    state.setLoading(true);
                    state.setError(null);
                    try {
                      final result = await api.predict(
                        features: Map<String, num>.from(state.features),
                        audience: state.audience,
                        includeExplanations: true,
                      );
                      state.setResult(result);
                      if (!context.mounted) return;
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => state.audience == 'doctor'
                              ? const DoctorScreen()
                              : const PatientScreen(),
                        ),
                      );
                    } catch (err) {
                      state.setError(
                        'Could not reach API. Start FastAPI on the host and check API_BASE_URL. ($err)',
                      );
                    } finally {
                      state.setLoading(false);
                    }
                  },
            child: Text(state.loading ? 'Running…' : 'Get screening result'),
          ),
        ],
      ),
    );
  }
}

class _SliderField extends StatelessWidget {
  const _SliderField({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.onChanged,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('$label: ${value.toStringAsFixed(0)}'),
        Slider(
          value: value.clamp(min, max),
          min: min,
          max: max,
          divisions: divisions,
          label: value.toStringAsFixed(0),
          onChanged: onChanged,
        ),
      ],
    );
  }
}
