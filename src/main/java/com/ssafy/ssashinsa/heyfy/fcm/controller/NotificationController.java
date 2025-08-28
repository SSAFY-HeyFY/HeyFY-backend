package com.ssafy.ssashinsa.heyfy.fcm.controller;

import com.google.firebase.messaging.FirebaseMessagingException;
import com.ssafy.ssashinsa.heyfy.fcm.docs.NotificationAllSendDocs;
import com.ssafy.ssashinsa.heyfy.fcm.docs.NotificationSendDocs;
import com.ssafy.ssashinsa.heyfy.fcm.docs.NotificationTag;
import com.ssafy.ssashinsa.heyfy.fcm.dto.BroadcastRequest;
import com.ssafy.ssashinsa.heyfy.fcm.dto.NotificationRequest;
import com.ssafy.ssashinsa.heyfy.fcm.service.FcmService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

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
        try {
            fcmService.sendNotification(request.getToken(), request.getTitle(), request.getBody());
            return ResponseEntity.ok("알림이 성공적으로 전송되었습니다.");
        } catch (FirebaseMessagingException e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("알림 전송에 실패했습니다: " + e.getMessage());
        }
    }

    @NotificationAllSendDocs
    @PostMapping("/send-notification/all")
    public ResponseEntity<String> sendNotificationToAll(@RequestBody BroadcastRequest request) {
        try {
            fcmService.sendNotificationToAll(request.getTitle(), request.getBody());
            return ResponseEntity.ok("모든 사용자에게 알림이 성공적으로 전송되었습니다.");
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("전체 알림 전송에 실패했습니다: " + e.getMessage());
        }
    }
}
