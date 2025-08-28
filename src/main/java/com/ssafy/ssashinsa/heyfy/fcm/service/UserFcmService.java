package com.ssafy.ssashinsa.heyfy.fcm.service;

import com.google.firebase.messaging.FirebaseMessagingException;
import com.google.firebase.messaging.MessagingErrorCode;
import com.ssafy.ssashinsa.heyfy.fcm.domain.FcmToken;
import com.ssafy.ssashinsa.heyfy.fcm.repository.FcmTokenRepository;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional
@RequiredArgsConstructor
public class UserFcmService {
    private final FcmTokenRepository fcmTokenRepository;
    private final UserRepository userRepository;

    public void registerToken(Long userId, String token) {
        Users user = userRepository.findById(userId)
                .orElseThrow(() -> new IllegalArgumentException("User not found"));

        if (!fcmTokenRepository.existsByToken(token)) {
            fcmTokenRepository.save(new FcmToken(user, token));
        } else {
            fcmTokenRepository.findByToken(token).ifPresent(t -> {
                if (!t.getUser().getId().equals(userId)) {
                    fcmTokenRepository.deleteByToken(token);
                    fcmTokenRepository.save(new FcmToken(user, token));
                }
            });
        }
    }

    public void deleteByToken(String token) {
        fcmTokenRepository.deleteByToken(token);
    }

    public void handleSendFailure(String token, FirebaseMessagingException e) {
        MessagingErrorCode code = e.getMessagingErrorCode();
        if (code == MessagingErrorCode.UNREGISTERED || code == MessagingErrorCode.INVALID_ARGUMENT) {
            fcmTokenRepository.deleteByToken(token);
        }
    }
}
