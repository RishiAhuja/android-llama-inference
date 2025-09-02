import 'package:flutter/material.dart';
import '../services/model_manager.dart';
import '../models/model_config.dart';

class ModelManagementScreen extends StatefulWidget {
  const ModelManagementScreen({super.key});

  @override
  State<ModelManagementScreen> createState() => _ModelManagementScreenState();
}

class _ModelManagementScreenState extends State<ModelManagementScreen> {
  final ModelManager _modelManager = ModelManager();
  bool _isLoading = false;
  List<String> _downloadedModels = [];
  Map<String, double> _downloadProgress = {};
  Map<String, bool> _isDownloading = {};

  @override
  void initState() {
    super.initState();
    _checkModelStatus();
  }

  Future<void> _checkModelStatus() async {
    setState(() => _isLoading = true);
    _downloadedModels = await _modelManager.getDownloadedModels();
    setState(() => _isLoading = false);
  }

  Future<void> _downloadModel(ModelConfig model) async {
    setState(() {
      _isDownloading[model.id] = true;
      _downloadProgress[model.id] = 0.0;
    });

    await _modelManager.downloadModel(
      model.id,
      onProgress: (progress) {
        setState(() {
          _downloadProgress[model.id] = progress;
        });
      },
      onError: (error) {
        setState(() {
          _isDownloading[model.id] = false;
          _downloadProgress.remove(model.id);
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Download failed: $error')),
          );
        }
      },
    );

    setState(() {
      _isDownloading[model.id] = false;
      _downloadProgress.remove(model.id);
    });
    _checkModelStatus();
  }

  Future<void> _deleteModel(ModelConfig model) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Delete Model'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Are you sure you want to delete "${model.name}"?'),
              const SizedBox(height: 8),
              Text(
                'Size: ${model.sizeInMB}MB',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 8),
              const Text(
                'This action cannot be undone.',
                style: TextStyle(fontWeight: FontWeight.w500),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () => Navigator.of(context).pop(true),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red,
                foregroundColor: Colors.white,
              ),
              child: const Text('Delete'),
            ),
          ],
        );
      },
    );

    if (confirmed == true) {
      await _modelManager.deleteModel(model.id);
      _checkModelStatus();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Model "${model.name}" deleted successfully')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Model Management'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _checkModelStatus,
              child: ListView.builder(
                padding: const EdgeInsets.all(16.0),
                itemCount: AvailableModels.models.length + 1,
                itemBuilder: (context, index) {
                  if (index == 0) {
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 16.0),
                      child: Card(
                        color: Theme.of(context).colorScheme.surfaceContainer,
                        child: Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(
                                    Icons.info_outline,
                                    color: Theme.of(context).colorScheme.primary,
                                  ),
                                  const SizedBox(width: 8),
                                  Text(
                                    'Available Models',
                                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'Download models to use for on-device inference. All models are under 1B parameters for optimal mobile performance.',
                                style: Theme.of(context).textTheme.bodyMedium,
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  }

                  final model = AvailableModels.models[index - 1];
                  final isDownloaded = _downloadedModels.contains(model.fileName);
                  final isDownloading = _isDownloading[model.id] ?? false;
                  final progress = _downloadProgress[model.id] ?? 0.0;

                  return _buildModelCard(model, isDownloaded, isDownloading, progress);
                },
              ),
            ),
    );
  }

  Widget _buildModelCard(ModelConfig model, bool isDownloaded, bool isDownloading, double progress) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12.0),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Text(
                            model.name,
                            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          if (model.isRecommended) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(
                                color: Colors.green.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(12),
                                border: Border.all(color: Colors.green.withOpacity(0.3)),
                              ),
                              child: Text(
                                'Recommended',
                                style: TextStyle(
                                  fontSize: 10,
                                  color: Colors.green[700],
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                          ],
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(
                        model.description,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Size: ${model.sizeInMB}MB',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Theme.of(context).colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
                _buildStatusIcon(isDownloaded, isDownloading),
              ],
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: model.capabilities.map((capability) => Chip(
                label: Text(
                  capability,
                  style: const TextStyle(fontSize: 11),
                ),
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                visualDensity: VisualDensity.compact,
              )).toList(),
            ),
            const SizedBox(height: 12),
            if (isDownloading) ...[
              LinearProgressIndicator(value: progress),
              const SizedBox(height: 8),
              Text('${(progress * 100).toInt()}% downloaded'),
              const SizedBox(height: 8),
            ],
            _buildActionButtons(model, isDownloaded, isDownloading),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusIcon(bool isDownloaded, bool isDownloading) {
    if (isDownloading) {
      return const SizedBox(
        width: 24,
        height: 24,
        child: CircularProgressIndicator(strokeWidth: 2),
      );
    } else if (isDownloaded) {
      return const Icon(Icons.check_circle, color: Colors.green, size: 24);
    } else {
      return const Icon(Icons.cloud_download, color: Colors.grey, size: 24);
    }
  }

  Widget _buildActionButtons(ModelConfig model, bool isDownloaded, bool isDownloading) {
    if (isDownloading) {
      return const SizedBox(
        height: 36,
        child: Center(
          child: Text('Downloading...'),
        ),
      );
    }

    if (isDownloaded) {
      return Row(
        children: [
          Expanded(
            child: ElevatedButton.icon(
              onPressed: () {
                _modelManager.setCurrentModel(model.id);
                Navigator.pushReplacementNamed(
                  context, 
                  '/chat',
                  arguments: {'modelId': model.id},
                );
              },
              icon: const Icon(Icons.chat),
              label: const Text('Use Model'),
            ),
          ),
          const SizedBox(width: 8),
          OutlinedButton.icon(
            onPressed: () => _deleteModel(model),
            icon: const Icon(Icons.delete, size: 18),
            label: const Text('Delete'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.red,
            ),
          ),
        ],
      );
    } else {
      return SizedBox(
        width: double.infinity,
        child: ElevatedButton.icon(
          onPressed: () => _downloadModel(model),
          icon: const Icon(Icons.download),
          label: const Text('Download'),
        ),
      );
    }
  }
}
