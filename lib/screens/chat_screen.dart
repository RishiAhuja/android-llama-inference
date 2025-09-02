import 'package:flutter/material.dart';
import '../models/chat_message.dart';
import '../services/llama_service.dart';
import '../services/model_manager.dart';

class ChatScreen extends StatefulWidget {
  final String? modelId;
  
  const ChatScreen({super.key, this.modelId});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final LlamaService _llamaService = LlamaService();
  final ModelManager _modelManager = ModelManager();
  final TextEditingController _promptController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<ChatMessage> _messages = [];

  bool _isLoading = true;
  bool _isGenerating = false;

  @override
  void initState() {
    super.initState();
    _initializeChat();
  }

  Future<void> _initializeChat() async {
    setState(() {
      _isLoading = true;
      _messages.add(ChatMessage(text: "Initializing...", isUser: false));
    });

    // Use the specified model ID or check for any available model
    await _modelManager.checkModelStatus(widget.modelId);

    if (_modelManager.status != ModelStatus.downloaded) {
      setState(() {
        _messages.last = ChatMessage(
          text:
              "Model not found. Please go to Model Management to download it.",
          isUser: false,
        );
        _isLoading = false;
      });
      return;
    }

    setState(() {
      _messages.last = ChatMessage(
        text: "Loading ${_modelManager.currentModel?.name ?? 'model'}...", 
        isUser: false,
      );
    });

    final success =
        await _llamaService.loadModel(_modelManager.modelFile!.path);

    setState(() {
      if (success) {
        _messages.last = ChatMessage(
          text: "Hello! I'm ${_modelManager.currentModel?.name ?? 'your AI assistant'}. How can I help you today?",
          isUser: false,
        );
      } else {
        _messages.last = ChatMessage(
          text: "Failed to load ${_modelManager.currentModel?.name ?? 'model'}. Please try restarting the app.",
          isUser: false,
        );
      }
      _isLoading = false;
    });
  }

  Future<void> _sendMessage() async {
    if (_promptController.text.trim().isEmpty ||
        _isGenerating ||
        !_llamaService.isInitialized) {
      return;
    }

    final prompt = _promptController.text.trim();
    _promptController.clear();

    setState(() {
      _messages.add(ChatMessage(text: prompt, isUser: true));
      _messages.add(ChatMessage(text: "", isUser: false)); // Empty message for typing indicator
      _isGenerating = true;
    });

    _scrollToBottom();

    final stopwatch = Stopwatch()..start();

    try {
      final response = await _llamaService.generateResponse(prompt);
      stopwatch.stop();
      
      if (mounted) {
        setState(() {
          // Replace the empty message with the actual response including timing
          _messages.last = ChatMessage(
            text: response, 
            isUser: false,
            responseTime: stopwatch.elapsed,
          );
        });
      }
    } catch (e) {
      stopwatch.stop();
      if (mounted) {
        setState(() {
          // Replace the empty message with the error
          _messages.last = ChatMessage(
            text: "Error: $e", 
            isUser: false,
            responseTime: stopwatch.elapsed,
          );
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isGenerating = false;
        });
        _scrollToBottom();
      }
    }
  }

  void _resetConversation() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Reset Conversation'),
          content: const Text('This will clear the chat history and reset the conversation context. Continue?'),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                setState(() {
                  _messages.clear();
                  _messages.add(ChatMessage(
                    text: "Conversation reset! Ask me anything.",
                    isUser: false,
                  ));
                });
                _llamaService.resetConversation();
                _scrollToBottom();
              },
              child: const Text('Reset'),
            ),
          ],
        );
      },
    );
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  void dispose() {
    _llamaService.dispose();
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gemma Chat'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          if (_llamaService.isInitialized && !_isLoading)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _resetConversation,
              tooltip: 'Reset Conversation',
            ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.pushNamed(context, '/models'),
            tooltip: 'Model Management',
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                return _MessageBubble(
                  message: _messages[index],
                  isTyping: _isGenerating && 
                           index == _messages.length - 1 && 
                           !_messages[index].isUser &&
                           _messages[index].text.isEmpty,
                );
              },
            ),
          ),
          if (_isGenerating)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Row(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 16),
                  Text("Generating response..."),
                ],
              ),
            ),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              border: Border(
                top: BorderSide(
                  color: Theme.of(context).colorScheme.outline,
                  width: 0.5,
                ),
              ),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _promptController,
                    decoration: InputDecoration(
                      hintText: 'Type your message...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(24),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                    maxLines: null,
                  ),
                ),
                const SizedBox(width: 8),
                FloatingActionButton(
                  onPressed: (_isLoading ||
                          _isGenerating ||
                          !_llamaService.isInitialized)
                      ? null
                      : _sendMessage,
                  mini: true,
                  child: const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _MessageBubble extends StatefulWidget {
  final ChatMessage message;
  final bool isTyping;

  const _MessageBubble({
    required this.message,
    this.isTyping = false,
  });

  @override
  State<_MessageBubble> createState() => _MessageBubbleState();
}

class _MessageBubbleState extends State<_MessageBubble>
    with TickerProviderStateMixin {
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    if (widget.isTyping) {
      _animationController.repeat();
    }
  }

  @override
  void didUpdateWidget(_MessageBubble oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isTyping && !oldWidget.isTyping) {
      _animationController.repeat();
    } else if (!widget.isTyping && oldWidget.isTyping) {
      _animationController.stop();
    }
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: widget.message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        decoration: BoxDecoration(
          color: widget.message.isUser
              ? Theme.of(context).colorScheme.primary
              : Theme.of(context).colorScheme.secondaryContainer,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            if (widget.isTyping)
              _buildTypingIndicator()
            else
              SelectableText(
                widget.message.text,
                style: TextStyle(
                  color: widget.message.isUser
                      ? Theme.of(context).colorScheme.onPrimary
                      : Theme.of(context).colorScheme.onSecondaryContainer,
                ),
              ),
            if (!widget.message.isUser && widget.message.responseTime != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  'Response time: ${_formatDuration(widget.message.responseTime!)}',
                  style: TextStyle(
                    fontSize: 11,
                    color: (widget.message.isUser
                            ? Theme.of(context).colorScheme.onPrimary
                            : Theme.of(context).colorScheme.onSecondaryContainer)
                        .withOpacity(0.7),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTypingIndicator() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Thinking',
          style: TextStyle(
            color: Theme.of(context).colorScheme.onSecondaryContainer.withOpacity(0.7),
            fontStyle: FontStyle.italic,
          ),
        ),
        const SizedBox(width: 8),
        SizedBox(
          width: 20,
          height: 20,
          child: AnimatedBuilder(
            animation: _animationController,
            builder: (context, child) {
              return Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: List.generate(3, (index) {
                  final delay = index * 0.2;
                  final animValue = (_animationController.value - delay).clamp(0.0, 1.0);
                  return Transform.translate(
                    offset: Offset(0, -5 * (1 - (animValue * 2 - 1).abs())),
                    child: Container(
                      width: 4,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.onSecondaryContainer.withOpacity(0.7),
                        shape: BoxShape.circle,
                      ),
                    ),
                  );
                }),
              );
            },
          ),
        ),
      ],
    );
  }

  String _formatDuration(Duration duration) {
    if (duration.inSeconds >= 60) {
      final minutes = duration.inMinutes;
      final seconds = duration.inSeconds % 60;
      return '${minutes}m ${seconds}s';
    } else if (duration.inSeconds >= 1) {
      return '${duration.inSeconds}.${(duration.inMilliseconds % 1000) ~/ 100}s';
    } else {
      return '${duration.inMilliseconds}ms';
    }
  }
}
