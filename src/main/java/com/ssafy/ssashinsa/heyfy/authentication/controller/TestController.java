package com.ssafy.ssashinsa.heyfy.authentication.controller;

import com.ssafy.ssashinsa.heyfy.authentication.dto.test.UserInfoDto;
import com.ssafy.ssashinsa.heyfy.authentication.service.AuthService;
import com.ssafy.ssashinsa.heyfy.authentication.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.common.dto.MessageDto;
import com.ssafy.ssashinsa.heyfy.swagger.response.ApiProtected;
import com.ssafy.ssashinsa.heyfy.swagger.response.ApiPublic;
import com.ssafy.ssashinsa.heyfy.swagger.response.ApiUserInfo;
import com.ssafy.ssashinsa.heyfy.swagger.response.ErrorsCommon;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

//테스트용 컨트롤러. 나중에 지울 예정
@RestController
@RequestMapping("/api/test")
@RequiredArgsConstructor
@ErrorsCommon
public class TestController {

    private final AuthService authService;
    @ApiPublic
    @GetMapping("/public")
    public ResponseEntity<MessageDto> publicEndpoint() {
        String message = "여기는 인증 없이 접근할 수 있는 공개 엔드포인트입니다.";
        return ResponseEntity.ok(new MessageDto(message));
    }

    @ApiProtected
    @GetMapping("/protected")
    public ResponseEntity<MessageDto> protectedEndpoint() {
        String username = SecurityUtil.getCurrentUsername();
        String message = "안녕하세요, " + username + "님! JWT 토큰이 있어야만 접근 가능한 보호된 엔드포인트입니다.";
        return ResponseEntity.ok(new MessageDto(message));
    }

    @ApiUserInfo
    @GetMapping("/userInfo")
    public ResponseEntity<UserInfoDto> getUserInfo() {
        String username = SecurityUtil.getCurrentUsername();
        String userKey = authService.getCurrentUserKey();

        UserInfoDto userInfo = new UserInfoDto(username, userKey);

        return ResponseEntity.ok(userInfo);
    }
}