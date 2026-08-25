import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../state/screening_state.dart';

class PatientScreen extends StatelessWidget {
  const PatientScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = context.watch<ScreeningState>();
    final result = state.lastResult;
    final risk = result?['risk_percent'];
    final disclaimer = result?['disclaimer'] as String? ??
        'Screening aid only — not a diagnosis.';
    final explanations = result?['explanations'] as Map<String, dynamic>?;
    final safe = (explanations?['safe_explanations'] as List?) ?? const [];

    return Scaffold(
      appBar: AppBar(title: const Text('Your screening summary')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Text(
            risk == null ? 'No result yet — run the questionnaire.' : 'Estimated risk score',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          if (risk != null) ...[
            const SizedBox(height: 8),
            Text(
              '${(risk as num).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.displayMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: Theme.of(context).colorScheme.primary,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              'This percentage is a model estimate for screening discussion, '
              'not a diagnosis or certainty of disease.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
          const SizedBox(height: 20),
          Text('What the model relied on', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          if (safe.isEmpty)
            const Text('Explanations unavailable for this session.')
          else
            ...safe.take(5).map((item) {
              final map = item as Map<String, dynamic>;
              return ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text('${map['feature']}'),
                subtitle: Text('${map['patient_safe_description']}'),
              );
            }),
          const Divider(height: 32),
          Text(disclaimer, style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}
