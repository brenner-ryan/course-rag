# Practical Pigeon Communication

instructor: Prof. Aldous Finch
link: https://example.edu/courses/pigeon-communication

## Lesson 1: Message Encoding and Payload Limits

A carrier pigeon has a practical payload of around seventy five grams, which in paper terms
is roughly ten sheets. Messages must therefore be encoded compactly. The traditional
approach is a fixed field format where each field has a known position and length, so no
delimiters are needed.

Compression helps but has an important caveat: a corrupted compressed message is usually
unrecoverable in full, whereas a corrupted fixed field message loses only the damaged
field. For this reason critical messages are sent uncompressed and redundantly, on two
birds, rather than compressed on one.

## Lesson 2: Routing and Delivery Guarantees

Pigeon routing offers at-most-once delivery and no acknowledgement. The bird either arrives
or it does not, and the sender learns nothing either way. Systems built on pigeon transport
therefore cannot assume delivery and must either tolerate loss or build acknowledgement out
of a second bird flying the reverse route.

Latency is highly variable, ranging from forty minutes to several days depending on
weather, predation and the bird's own judgement. Designers should treat the transport as
eventually consistent at best, and should never use it for anything requiring ordering
between messages, since two birds released together routinely arrive days apart.

## Lesson 3: Security Considerations

A pigeon is an unauthenticated transport. Anyone who intercepts the bird can read the
message, alter it, and release the bird to continue its journey, and the recipient has no
way to detect this. Encryption of the payload is the only real protection, and it must be
done before the message is attached.

Physical security of the loft matters more than most operators expect, because an attacker
who gains access to the loft can substitute their own birds and receive all future traffic.
Signs of compromise include unfamiliar birds, and messages arriving substantially faster
than the route normally allows.
