import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../state/screening_state.dart';

class DoctorScreen extends StatelessWidget {
  const DoctorScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = context.watch<ScreeningState>();
    final result = state.lastResult;
    final explanations = result?['explanations'] as Map<String, dynamic>?;
    final agreement = explanations?['agreement'] as Map<String, dynamic>?;

    return Scaffold(
      appBar: AppBar(title: const Text('Clinician view')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Text('Model output', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 8),
          if (result == null)
            const Text('No result yet — run the questionnaire.')
          else ...[
            _kv('Risk %', '${result['risk_percent']}'),
            _kv('Probability', '${result['risk_probability']}'),
            _kv('Predicted label', '${result['predicted_label']}'),
            _kv('Threshold', '${result['threshold']}'),
            _kv('Model', '${result['model_name']}'),
            _kv('Model version', '${result['model_version']}'),
            const SizedBox(height: 16),
            Text('Explanation concordance', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (agreement == null)
              const Text('Agreement unavailable.')
            else ...[
              _kv('Top-k', '${agreement['top_k']}'),
              _kv('Exact set match', '${agreement['exact_top_k_set_match']}'),
              _kv('Jaccard', '${agreement['jaccard_overlap']}'),
              Text(
                '${agreement['note']}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            const SizedBox(height: 16),
            Text(
              'Reminder: concordance is not predictive confidence. '
              'Attributions are not causal. This is not a diagnostic device.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const Divider(height: 32),
            Text('${result['disclaimer']}', style: Theme.of(context).textTheme.bodySmall),
          ],
        ],
      ),
    );
  }

  Widget _kv(String k, String v) => Padding(
        padding: const EdgeInsets.only(bottom: 6),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(width: 140, child: Text(k, style: const TextStyle(fontWeight: FontWeight.w600))),
            Expanded(child: Text(v)),
          ],
        ),
      );
}
