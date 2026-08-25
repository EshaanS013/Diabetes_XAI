import 'package:flutter/material.dart';

import 'doctor_screen.dart';
import 'patient_screen.dart';
import 'questionnaire_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  static const disclaimer =
      'This app is a research screening aid. It does not diagnose diabetes, '
      'does not replace a clinician, and explanations are associations—not causes.';

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      body: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              scheme.surface,
              scheme.primary.withOpacity(0.08),
              const Color(0xFFE8F3EF),
            ],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(24, 28, 24, 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Diabetes XAI',
                  style: Theme.of(context).textTheme.displaySmall?.copyWith(
                        fontWeight: FontWeight.w700,
                        color: scheme.primary,
                      ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Explainable risk screening',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 16),
                Text(
                  disclaimer,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: scheme.onSurface.withOpacity(0.75),
                      ),
                ),
                const Spacer(),
                _HomeButton(
                  label: 'Start questionnaire',
                  onPressed: () => Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const QuestionnaireScreen()),
                  ),
                ),
                const SizedBox(height: 12),
                _HomeButton(
                  label: 'Patient result view (demo)',
                  outlined: true,
                  onPressed: () => Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const PatientScreen()),
                  ),
                ),
                const SizedBox(height: 12),
                _HomeButton(
                  label: 'Doctor result view (demo)',
                  outlined: true,
                  onPressed: () => Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const DoctorScreen()),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _HomeButton extends StatelessWidget {
  const _HomeButton({
    required this.label,
    required this.onPressed,
    this.outlined = false,
  });

  final String label;
  final VoidCallback onPressed;
  final bool outlined;

  @override
  Widget build(BuildContext context) {
    if (outlined) {
      return SizedBox(
        width: double.infinity,
        child: OutlinedButton(onPressed: onPressed, child: Text(label)),
      );
    }
    return SizedBox(
      width: double.infinity,
      child: FilledButton(onPressed: onPressed, child: Text(label)),
    );
  }
}
