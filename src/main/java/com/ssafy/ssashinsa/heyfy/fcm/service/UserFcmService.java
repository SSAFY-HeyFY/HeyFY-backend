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
            // 다른 계정에 묶인 동일 토큰이 재등록되는 케이스 처리(선택)
            fcmTokenRepository.findByToken(token).ifPresent(t -> {
                if (!t.getUser().getId().equals(userId)) {
                    fcmTokenRepository.deleteByToken(token);
                    fcmTokenRepository.save(new FcmToken(user, token));
                }
            });
        }
    }

    /** 비인증 public 엔드포인트가 호출할 삭제 로직 */
    public void deleteByToken(String token) {
        fcmTokenRepository.deleteByToken(token);
    }

    /** 발송 실패 시: UNREGISTERED/INVALID_ARGUMENT → 즉시 삭제 */
    public void handleSendFailure(String token, FirebaseMessagingException e) {
        MessagingErrorCode code = e.getMessagingErrorCode();
        if (code == MessagingErrorCode.UNREGISTERED || code == MessagingErrorCode.INVALID_ARGUMENT) {
            fcmTokenRepository.deleteByToken(token);
        }
    }
}
