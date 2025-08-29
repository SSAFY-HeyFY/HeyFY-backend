package com.ssafy.ssashinsa.heyfy.authentication.controller;

import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthRefreshDocs;
import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthSignInDocs;
import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthSignUpDocs;
import com.ssafy.ssashinsa.heyfy.authentication.docs.SidRefreshDocs;
import com.ssafy.ssashinsa.heyfy.authentication.dto.*;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.authentication.service.AuthService;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@Tag(name = "계정 관리", description = "계정 관리 API")
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @AuthSignInDocs
    @PostMapping("/signin")
    public ResponseEntity<SignInSuccessDto> signIn(@RequestBody SignInDto signInDto) {
        return ResponseEntity.ok(authService.signIn(signInDto));
    }

    @AuthSignUpDocs
    @PostMapping("/signup")
    public ResponseEntity<SignUpSuccessDto> signUp(@Valid @RequestBody SignUpDto signUpDto){
        return ResponseEntity.ok(authService.signUp(signUpDto));
    }

    @AuthRefreshDocs
    @PostMapping("/refresh")
    public ResponseEntity<TokenDto> refreshAccessToken(@RequestHeader("Authorization") String authorizationHeader, @RequestHeader("RefreshToken") String refreshToken) {
        return ResponseEntity.ok(authService.refreshAccessToken(authorizationHeader, refreshToken));
    }

    @SidRefreshDocs
    @PostMapping("/sid/refresh")
    public ResponseEntity<SidDto> issueSid(@RequestBody SecondaryAuthRequestDto requestDto) {
        String newSid = "";
        try {
            newSid = authService.issueSid(requestDto.getPinNumber());
        }catch(CustomException e){
            if(e.getErrorCode().equals(AuthErrorCode.INVALID_PIN_NUMBER)){
                return ResponseEntity.ok(new SidDto(newSid, false));
            }
            throw e;
        }
        return ResponseEntity.ok(new SidDto(newSid, true));
    }
}