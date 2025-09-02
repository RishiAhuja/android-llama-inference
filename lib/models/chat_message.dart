class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final Duration? responseTime;  // Track how long it took to generate the response

  ChatMessage({
    required this.text,
    required this.isUser,
    DateTime? timestamp,
    this.responseTime,
  }) : timestamp = timestamp ?? DateTime.now();
}
