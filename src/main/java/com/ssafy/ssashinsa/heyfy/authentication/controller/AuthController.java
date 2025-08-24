package com.ssafy.ssashinsa.heyfy.authentication.controller;

import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthRefreshDocs;
import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthSignInDocs;
import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthSignUpDocs;
import com.ssafy.ssashinsa.heyfy.authentication.dto.*;
import com.ssafy.ssashinsa.heyfy.authentication.dto.test.MessageDto;
import com.ssafy.ssashinsa.heyfy.authentication.service.AuthService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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

    @PostMapping("/txntoken")
    public ResponseEntity<TxnAuthTokenDto> issueTxnAuthToken() {
        String token = authService.createTxnAuthToken();
        return ResponseEntity.ok(new TxnAuthTokenDto(token));
    }

    @PostMapping("/verifypin")
    public ResponseEntity<MessageDto> verifyPinAndTxnToken(
                                                            @RequestHeader("TxnAuthToken") String txnAuthToken,
                                                            @RequestBody TxnAuthRequestDto requestDto
    ) {
        authService.verifySecondaryAuth(requestDto.getPinNumber(), txnAuthToken);
        return ResponseEntity.ok(new MessageDto("2차 인증이 성공적으로 완료되었습니다."));
    }

}