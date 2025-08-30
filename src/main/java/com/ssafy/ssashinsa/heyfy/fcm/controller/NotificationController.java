package com.ssafy.ssashinsa.heyfy.fcm.controller;

import com.google.firebase.messaging.FirebaseMessagingException;
import com.ssafy.ssashinsa.heyfy.fcm.docs.NotificationAllSendDocs;
import com.ssafy.ssashinsa.heyfy.fcm.docs.NotificationSendDocs;
import com.ssafy.ssashinsa.heyfy.fcm.docs.NotificationTag;
import com.ssafy.ssashinsa.heyfy.fcm.dto.BroadcastRequest;
import com.ssafy.ssashinsa.heyfy.fcm.dto.NotificationRequest;
import com.ssafy.ssashinsa.heyfy.fcm.service.FcmService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@NotificationTag
public class NotificationController {

    private final FcmService fcmService;

    public NotificationController(FcmService fcmService) {
        this.fcmService = fcmService;
    }

    @NotificationSendDocs
    @PostMapping("/send-notification")
    public ResponseEntity<String> sendNotification(@RequestBody NotificationRequest request) {
        log.info("알림 보내기 요청: 토큰={}, 제목={}, 내용={}", request.getToken(), request.getTitle(), request.getBody());
        try {
            fcmService.sendNotification(request.getToken(), request.getTitle(), request.getBody());
            return ResponseEntity.ok("알림이 성공적으로 전송되었습니다.");
        } catch (FirebaseMessagingException e) {
            log.info("알림 전송 실패: {}", e.getMessage());
            return ResponseEntity.status(500).body("알림 전송에 실패했습니다: " + e.getMessage());
        }
    }

    @NotificationAllSendDocs
    @PostMapping("/send-notification/all")
    public ResponseEntity<String> sendNotificationToAll(@RequestBody BroadcastRequest request) {
        log.info("전체 알림 보내기 요청: 제목={}, 내용={}", request.getTitle(), request.getBody());
        try {
            fcmService.sendNotificationToAll(request.getTitle(), request.getBody());
            return ResponseEntity.ok("모든 사용자에게 알림이 성공적으로 전송되었습니다.");
        } catch (Exception e) {
            log.info("전체 알림 전송 실패: {}", e.getMessage());
            return ResponseEntity.status(500).body("전체 알림 전송에 실패했습니다: " + e.getMessage());
        }
    }
}
