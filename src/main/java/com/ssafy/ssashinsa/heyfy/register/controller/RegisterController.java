package com.ssafy.ssashinsa.heyfy.register.controller;

import com.ssafy.ssashinsa.heyfy.register.docs.CreateDepositAccountDocs;
import com.ssafy.ssashinsa.heyfy.register.dto.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.register.service.RegisterService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/register")
@RequiredArgsConstructor
public class RegisterController {

    private final RegisterService registerService;


    @CreateDepositAccountDocs
    @PostMapping("/createdeposit")
    public ResponseEntity<ShinhanCreateDepositResponseDto> createDepositAccount() {

        ShinhanCreateDepositResponseDto responseDto = registerService.createDepositAccount();

        return ResponseEntity.ok(responseDto);
    }
}
