package com.ssafy.ssashinsa.heyfy.fcm.controller;

import com.ssafy.ssashinsa.heyfy.authentication.jwt.CustomUserDetails; // CustomUserDetails 임포트
import com.ssafy.ssashinsa.heyfy.fcm.docs.FcmDeletePublicDocs;
import com.ssafy.ssashinsa.heyfy.fcm.docs.FcmRegisterDocs;
import com.ssafy.ssashinsa.heyfy.fcm.docs.FcmTag;
import com.ssafy.ssashinsa.heyfy.fcm.dto.FcmTokenRequest;
import com.ssafy.ssashinsa.heyfy.fcm.service.UserFcmService;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/users")
@FcmTag
public class UserFcmController {

    private final UserFcmService userFcmService;
    private final UserRepository userRepository;

    @FcmRegisterDocs
    @PostMapping("/fcm-token")
    public ResponseEntity<Void> registerFcmToken(@RequestBody FcmTokenRequest request) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();

        CustomUserDetails userDetails = (CustomUserDetails) authentication.getPrincipal();
        String currentStudentId = userDetails.getUsername();

        log.info("FCM 토큰 등록 요청. Student ID: {}", currentStudentId);

        Users currentUser = userRepository.findByStudentId(currentStudentId)
                .orElseThrow(() -> new IllegalArgumentException("User not found with studentId: " + currentStudentId));

        userFcmService.registerToken(currentUser.getId(), request.getFcmToken());

        return ResponseEntity.ok().build();
    }

    @FcmDeletePublicDocs
    @DeleteMapping("/tokens/public")
    public ResponseEntity<Void> deletePublic(@RequestBody FcmTokenRequest request) {
        userFcmService.deleteByToken(request.getFcmToken());
        return ResponseEntity.ok().build();
    }
}