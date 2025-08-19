package com.ssafy.ssashinsa.heyfy.authentication.controller;

import com.ssafy.ssashinsa.heyfy.authentication.dto.test.MessageDto;
import com.ssafy.ssashinsa.heyfy.authentication.service.AuthService;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import io.swagger.v3.oas.annotations.Hidden;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


//테스트용 컨트롤러. 나중에 지울 예정 - 마지막에 지울것
@Hidden
@RestController
@RequestMapping("/api/test")
@RequiredArgsConstructor
public class TestController {

    private final AuthService authService;
    @GetMapping("/public")
    public ResponseEntity<MessageDto> publicEndpoint() {
        String message = "여기는 인증 없이 접근할 수 있는 공개 엔드포인트입니다.";
        return ResponseEntity.ok(new MessageDto(message));
    }

    @GetMapping("/protected")
    public ResponseEntity<MessageDto> protectedEndpoint() {
        String studentId = SecurityUtil.getCurrentStudentId();
        String message = "안녕하세요, " + studentId + "님! JWT 토큰이 있어야만 접근 가능한 보호된 엔드포인트입니다.";
        return ResponseEntity.ok(new MessageDto(message));
    }

}

